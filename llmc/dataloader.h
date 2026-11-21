/*
Implements:
- DataLoader for model training. Reads and serves data shards.
- EvalLoader for multiple-choice evaluation datasets, e.g. HellaSwag.
*/
#ifndef DATALOADER_H
#define DATALOADER_H

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <assert.h>
#include <string.h>
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck
// defines: mallocCheck
#include "utils.h"
#include "rand.h"

// ----------------------------------------------------------------------------
// implementation of glob for Windows is in dev/unistd.h
#ifndef _WIN32
#include <glob.h>
#include <unistd.h> // access, sleep (used by the streaming mode below)
#else
#define sleep(s) Sleep((s) * 1000)
#endif
// ----------------------------------------------------------------------------
// Distributed Data Loader
#define HEADER_SIZE 256

typedef struct {
    // variables related to distributed training
    // each process/worker has to access different parts of the data
    int process_rank;
    int num_processes;
    // batch and token information
    size_t B;
    size_t T;
    size_t num_tokens; // total number of tokens
    size_t shard_num_samples;  // total number of samples in the current shard per process
    // shards and current position
    glob_t glob_result; // stores the result of glob, for all shards we want to iterate
    size_t current_shard_idx; // the current shard we are reading from
    size_t current_sample_idx; // the current sample we are reading from
    // file handle
    FILE* tokens_file;
    // data buffers
    uint16_t* buffer; // we fread data from file into this buffer
    int* inputs;  // input tokens into transformer
    int* targets; // target tokens for the transformer
    // random shuffle related variables
    mt19937_state shuffle_rng;
    int should_shuffle;
    int* shard_indices;
    int* intra_shard_indices;
    // sizes in bytes
    size_t total_batch_size_bytes;  // total across all processes
    size_t local_batch_offset_bytes;  // inner-sample offset for this process
    size_t header_bytes;  // header size in bytes
    int64_t file_size_bytes;
    // streaming mode: shards are still being written by a producer (e.g. dev/data/climbmix_stream.py)
    // when we run out of known shards we re-glob and wait for the next one instead of wrapping.
    // the producer is done when <dir of pattern>/STREAM_DONE exists or stream_expected_shards are present.
    // in streaming mode shards are visited in (sorted) filename order; samples within a shard are still shuffled.
    int stream;
    int stream_expected_shards; // 0 = unknown, rely on the STREAM_DONE marker
    char* filename_pattern;
    char* stream_done_path;
} DataLoader;

int64_t dataloader_load_shard_(DataLoader *loader, int shard_index) {
    if (loader->should_shuffle && !loader->stream) {
        shard_index = loader->shard_indices[shard_index];
    }
    // use the first glob match as the filename for now
    const char* filename = loader->glob_result.gl_pathv[shard_index];
    // open the input file for reading. also only a single file can be opened at a time
    if (loader->tokens_file != NULL) {
        fcloseCheck(loader->tokens_file);
    }
    loader->tokens_file = fopenCheck(filename, "rb");
    // validate the header
    int header[HEADER_SIZE];
    freadCheck(header, sizeof(int), HEADER_SIZE, loader->tokens_file);
    if (header[0] != 20240520) {
        printf("Bad magic in the data file\n");
        printf("---> HINT: Are you passing in a correct file?\n");
        printf("---> HINT: The data encoding may have changed, re-run data prepro or refer again to README.\n");
        exit(EXIT_FAILURE);
    }
    if (header[1] != 1) { printf("Bad version in data file\n"); exit(EXIT_FAILURE); }
    int64_t ntok = header[2]; // number of tokens in the file
    assert(ntok > 0); // we expect some tokens in the file. this should never trip, right?
    // determine the file size and make sure it is consistent with the number of tokens
    fseekCheck(loader->tokens_file, 0, SEEK_END); // seek to end of file
    loader->file_size_bytes = ftell(loader->tokens_file); // read the offset, i.e. file size
    fseekCheck(loader->tokens_file, 0, SEEK_SET); // seek back to the beginning
    // we expect ntok in the file to be consistent with filesize, assert that is the case
    int64_t expected_file_size = HEADER_SIZE * sizeof(int) + ntok * sizeof(uint16_t);
    if (loader->file_size_bytes != expected_file_size) {
        printf("Error: file size is not as expected\n");
        exit(EXIT_FAILURE);
    }
    // -1 uint16_t due to us taking B*T+1 tokens but moving by B*T tokens
    loader->shard_num_samples = (ntok * sizeof(uint16_t) - sizeof(uint16_t)) / loader->total_batch_size_bytes;
    return ntok;
}

// ----------------------------------------------------------------------------
// streaming helpers

int dataloader_stream_done_(DataLoader *loader) {
    if (loader->stream_expected_shards > 0 &&
        loader->glob_result.gl_pathc >= (size_t) loader->stream_expected_shards) { return 1; }
    return access(loader->stream_done_path, F_OK) == 0;
}

void dataloader_reglob_(DataLoader *loader) {
    // shard files are written to a temp name and atomically renamed, and filenames are
    // zero-padded, so a new glob only ever appends to the (sorted) list we already have
    glob_t fresh;
    memset(&fresh, 0, sizeof(fresh));
    if (glob(loader->filename_pattern, 0, NULL, &fresh) != 0) { globfree(&fresh); return; }
    if (fresh.gl_pathc <= loader->glob_result.gl_pathc) { globfree(&fresh); return; }
    globfree(&loader->glob_result);
    loader->glob_result = fresh;
    if (loader->should_shuffle) {
        // shard order is not permuted in streaming mode, but keep shard_indices valid (it is saved in the state)
        free(loader->shard_indices);
        loader->shard_indices = (int*)mallocCheck(loader->glob_result.gl_pathc * sizeof(int));
        init_identity_permutation(loader->shard_indices, (int) loader->glob_result.gl_pathc);
    }
}

// block until shard `shard_idx` exists. returns 1 if it is available, 0 if the producer is done without it
int dataloader_wait_for_shard_(DataLoader *loader, size_t shard_idx) {
    int polls = 0;
    while (1) {
        if (shard_idx < loader->glob_result.gl_pathc) { return 1; }
        // check the marker before globbing: the producer renames all shards before writing it
        int done = dataloader_stream_done_(loader);
        dataloader_reglob_(loader);
        if (shard_idx < loader->glob_result.gl_pathc) { return 1; }
        if (done) { return 0; }
        if (polls % 60 == 0 && loader->process_rank == 0) {
            printf("DataLoader: waiting for shard %zu of %s (%zu available so far)\n",
                   shard_idx, loader->filename_pattern, loader->glob_result.gl_pathc);
            fflush(stdout);
        }
        polls++;
        sleep(1);
    }
}

void prepare_intra_shard_indices_(DataLoader *loader) {
    // shuffle the examples inside the shards
    if (loader->intra_shard_indices != NULL) {
        // in case shards have different number of samples / sizes
        free(loader->intra_shard_indices);
    }
    loader->intra_shard_indices = (int*)mallocCheck(loader->shard_num_samples * sizeof(int));
    init_identity_permutation(loader->intra_shard_indices, (int) loader->shard_num_samples);
    random_permutation(loader->intra_shard_indices, (int) loader->shard_num_samples, &loader->shuffle_rng);
}

void dataloader_reset(DataLoader *loader) {
    loader->current_shard_idx = 0;
    loader->current_sample_idx = 0;

    if (loader->should_shuffle && !loader->stream) {  // shuffle the shards
        random_permutation(loader->shard_indices, (int) loader->glob_result.gl_pathc, &loader->shuffle_rng);
    }

    dataloader_load_shard_(loader, (int) loader->current_shard_idx);

    if (loader->should_shuffle) {
        prepare_intra_shard_indices_(loader);
    }
}

void dataloader_advance_(DataLoader *loader) {
    if (loader->current_shard_idx == loader->glob_result.gl_pathc - 1) {
        // streaming: the next shard may still be in production, wait for it instead of wrapping
        if (!loader->stream || !dataloader_wait_for_shard_(loader, loader->current_shard_idx + 1)) {
            // if we are at the last shard, we reset the loader and start a new epoch
            dataloader_reset(loader);
            return;
        }
    }

    // advance the loader by loading the next data shard and resetting the position
    loader->current_shard_idx = (loader->current_shard_idx + 1) % loader->glob_result.gl_pathc;
    loader->current_sample_idx = 0;
    dataloader_load_shard_(loader, (int) loader->current_shard_idx);

    if (loader->should_shuffle) {
        prepare_intra_shard_indices_(loader);
    }
}

void dataloader_init_(DataLoader *loader,
                      const char* filename_pattern,
                      size_t B,
                      size_t T,
                      int process_rank,
                      int num_processes,
                      int should_shuffle,
                      int stream,
                      int stream_expected_shards) {
    loader->process_rank = process_rank;
    loader->num_processes = num_processes;
    loader->B = B;
    loader->T = T;
    loader->tokens_file = NULL;
    loader->should_shuffle = should_shuffle;
    loader->header_bytes = HEADER_SIZE * sizeof(int);
    loader->total_batch_size_bytes = ((loader->num_processes * (loader->B * loader->T)) * sizeof(uint16_t));
    loader->local_batch_offset_bytes = loader->process_rank * loader->B * loader->T * sizeof(uint16_t);
    loader->stream = stream;
    loader->stream_expected_shards = stream_expected_shards;
    loader->filename_pattern = NULL;
    loader->stream_done_path = NULL;
    loader->shard_indices = NULL;
    memset(&loader->glob_result, 0, sizeof(glob_t));

    if (stream) {
        // remember the pattern, and look for the STREAM_DONE marker next to the shards
        size_t plen = strlen(filename_pattern);
        loader->filename_pattern = (char*)mallocCheck(plen + 1);
        strcpy(loader->filename_pattern, filename_pattern);
        const char* slash = strrchr(filename_pattern, '/');
        size_t dlen = slash ? (size_t)(slash - filename_pattern) : 1;
        loader->stream_done_path = (char*)mallocCheck(dlen + 16);
        if (slash) { memcpy(loader->stream_done_path, filename_pattern, dlen); }
        else { loader->stream_done_path[0] = '.'; }
        strcpy(loader->stream_done_path + dlen, "/STREAM_DONE");
        // wait for at least the first shard to show up
        if (!dataloader_wait_for_shard_(loader, 0)) {
            printf("Error: producer is done but no files match the pattern: %s\n", filename_pattern);
            exit(EXIT_FAILURE);
        }
    } else {
        // glob to get the list of files matching the pattern, these are our data shards
        int glob_status = glob(filename_pattern, 0, NULL, &loader->glob_result);
        if (glob_status != 0) {
            printf("Error: failed to glob pattern: %s\n", filename_pattern);
            exit(EXIT_FAILURE);
        }
        if (loader->glob_result.gl_pathc == 0) {
            printf("Error: no files found matching the pattern: %s\n", filename_pattern);
            exit(EXIT_FAILURE);
        }
    }

    if (should_shuffle) {
        mt19937_state shuffle_rng;
        manual_seed(&shuffle_rng, 42 + process_rank);
        loader->shuffle_rng = shuffle_rng;
        free(loader->shard_indices); // may have been allocated by the streaming re-glob
        loader->shard_indices = (int*)mallocCheck(loader->glob_result.gl_pathc * sizeof(int));
        init_identity_permutation(loader->shard_indices, (int) loader->glob_result.gl_pathc);
        loader->intra_shard_indices = NULL;  // dynamically allocated allowing different shard sizes
    }

    // inspect and validate all shards so we don't get any runtime errors later
    // if too slow / too many shards, may wish to revisit later
    int64_t ntok_total = 0;
    for (int shard_index = 0; shard_index < loader->glob_result.gl_pathc; shard_index++) {
        int64_t shard_ntok = dataloader_load_shard_(loader, shard_index);
        // we need at least one batch/shard, the way things are written right now.
        // can be relaxed a lot later.
        assert(shard_ntok >= (int64_t) (num_processes * B * T + 1));
        ntok_total += shard_ntok;
    }
    // debugging prints
    // printf("DataLoader: filename_pattern: %s\n", filename_pattern);
    // printf("DataLoader: Found %ld tokens across %zu shards\n", ntok_total, loader->glob_result.gl_pathc);

    // allocate all the space we'll need
    loader->buffer = (uint16_t*)mallocCheck((B * T + 1) * sizeof(uint16_t));
    loader->inputs = (int*)mallocCheck(B * T * sizeof(int));
    loader->targets = (int*)mallocCheck(B * T * sizeof(int));
    loader->num_tokens = ntok_total;

    // reset the loader, to initialize it
    dataloader_reset(loader);
}

void dataloader_init(DataLoader *loader,
                     const char* filename_pattern,
                     size_t B,
                     size_t T,
                     int process_rank,
                     int num_processes,
                     int should_shuffle) {
    dataloader_init_(loader, filename_pattern, B, T, process_rank, num_processes, should_shuffle, 0, 0);
}

// like dataloader_init, but the shards matching filename_pattern may still be in production:
// re-glob and wait for new shards instead of wrapping around to the first one (see DataLoader.stream)
void dataloader_init_streaming(DataLoader *loader,
                               const char* filename_pattern,
                               size_t B,
                               size_t T,
                               int process_rank,
                               int num_processes,
                               int should_shuffle,
                               int expected_shards) {
    dataloader_init_(loader, filename_pattern, B, T, process_rank, num_processes, should_shuffle, 1, expected_shards);
}

void dataloader_load_batch(DataLoader* loader) {
    assert(!loader->should_shuffle || (loader->should_shuffle && loader->intra_shard_indices != NULL));
    assert(loader->current_sample_idx < loader->shard_num_samples);
    size_t idx = loader->should_shuffle ? loader->intra_shard_indices[loader->current_sample_idx] : loader->current_sample_idx;
    size_t global_batch_offset_bytes = idx * loader->total_batch_size_bytes;
    int64_t current_offset = loader->header_bytes + global_batch_offset_bytes + loader->local_batch_offset_bytes;

    size_t B = loader->B;
    size_t T = loader->T;
    // read B*T+1 uint16_t tokens from the file into buffer
    fseekCheck(loader->tokens_file, (int) current_offset, SEEK_SET);
    freadCheck(loader->buffer, sizeof(uint16_t), B*T+1, loader->tokens_file);
    // decode the buffer into inputs and targets (cast to int)
    for (int i = 0; i < B*T; i++) {
        loader->inputs[i] = (int)loader->buffer[i];
        loader->targets[i] = (int)loader->buffer[i+1];
    }
}

void dataloader_next_batch(DataLoader *loader) {
    // if the next batch would go past the end of the file, advance the loader
    if (loader->current_sample_idx >= loader->shard_num_samples) {
        dataloader_advance_(loader);
    }
    dataloader_load_batch(loader);
    loader->current_sample_idx += 1;
}


void dataloader_resume(DataLoader *loader, size_t current_shard_idx, size_t current_sample_idx) {
    // used during model resumption (-y 1) flag
    if (loader->stream && !dataloader_wait_for_shard_(loader, current_shard_idx)) {
        printf("Error: cannot resume, shard %zu of %s does not exist\n", current_shard_idx, loader->filename_pattern);
        exit(EXIT_FAILURE);
    }
    loader->current_shard_idx = current_shard_idx;
    loader->current_sample_idx = current_sample_idx;
    dataloader_load_shard_(loader, (int) loader->current_shard_idx);
}

// Resume from a state saved with a different sample size (micro batch B, or number of processes).
// A "sample" is num_processes*B*T tokens, so a different B splits the shard into a different number of
// samples and the saved position / permutation no longer line up with the current loader.
// We convert the saved position into tokens of the current shard:
// - shuffled: every current sample that is mostly covered by already-consumed saved samples counts
//   as seen (exact half-covered ties alternate); seen samples go first in the new permutation, the rest follow in a fresh random order
//   (deterministic, and the loader's shuffle_rng is left untouched), and we resume right after the seen ones.
// - not shuffled: resume at round(saved_sample_idx * saved_sample_tokens / current_sample_tokens).
// At most a fraction of one sample's worth of tokens per boundary gets skipped or repeated.
// saved_perm / saved_num_samples may be NULL / 0 when the loader is not shuffled.
// call this after dataloader_resume() so the current shard (and its shard_num_samples) is loaded.
void dataloader_convert_position_(DataLoader *loader, size_t saved_sample_idx, size_t saved_sample_tokens,
                                  const int* saved_perm, size_t saved_num_samples) {
    size_t new_sample_tokens = loader->num_processes * loader->B * loader->T;
    size_t n_new = loader->shard_num_samples;
    if (!loader->should_shuffle) {
        size_t tokens = saved_sample_idx * saved_sample_tokens;
        size_t idx = (tokens + new_sample_tokens / 2) / new_sample_tokens;
        loader->current_sample_idx = idx > n_new ? n_new : idx;
        return;
    }
    if (saved_sample_idx > saved_num_samples) { saved_sample_idx = saved_num_samples; }
    // which saved samples were already consumed in this shard
    unsigned char* consumed = (unsigned char*)mallocCheck(saved_num_samples > 0 ? saved_num_samples : 1);
    memset(consumed, 0, saved_num_samples > 0 ? saved_num_samples : 1);
    for (size_t i = 0; i < saved_sample_idx; i++) {
        int k = saved_perm[i];
        if (k >= 0 && (size_t) k < saved_num_samples) { consumed[k] = 1; }
    }
    unsigned char* seen = (unsigned char*)mallocCheck(n_new > 0 ? n_new : 1);
    size_t num_seen = 0;
    int tie = 0;
    for (size_t j = 0; j < n_new; j++) {
        size_t start = j * new_sample_tokens, end = start + new_sample_tokens;
        size_t covered = 0;
        for (size_t i = start / saved_sample_tokens; i < saved_num_samples && i * saved_sample_tokens < end; i++) {
            if (!consumed[i]) { continue; }
            size_t s0 = i * saved_sample_tokens, s1 = s0 + saved_sample_tokens;
            size_t lo = s0 > start ? s0 : start, hi = s1 < end ? s1 : end;
            if (hi > lo) { covered += hi - lo; }
        }
        // more than half consumed counts as seen; exactly half (e.g. B 8 -> 16 with one of the two halves
        // consumed) alternates, so the skipped and repeated tokens roughly cancel out
        if (2 * covered != new_sample_tokens) { seen[j] = (2 * covered > new_sample_tokens); }
        else { seen[j] = (unsigned char) tie; tie ^= 1; }
        num_seen += seen[j];
    }
    // new permutation: seen samples first (they are skipped), then the unseen ones shuffled
    free(loader->intra_shard_indices);
    loader->intra_shard_indices = (int*)mallocCheck((n_new > 0 ? n_new : 1) * sizeof(int));
    size_t a = 0, b = num_seen;
    for (size_t j = 0; j < n_new; j++) {
        if (seen[j]) { loader->intra_shard_indices[a++] = (int) j; }
        else { loader->intra_shard_indices[b++] = (int) j; }
    }
    if (n_new > num_seen) {
        mt19937_state rng;
        manual_seed(&rng, (unsigned int) (1337 + 7919 * loader->current_shard_idx + loader->process_rank));
        random_permutation(loader->intra_shard_indices + num_seen, (int) (n_new - num_seen), &rng);
    }
    loader->current_sample_idx = num_seen;
    free(consumed);
    free(seen);
}

// resume from a state file: load shard current_shard_idx and restore the position in it.
// saved_num_processes / saved_B / saved_T describe the run that wrote the state (saved_B / saved_T may be 0
// for state files that predate storing them). saved_perm / saved_num_samples are the saved intra-shard
// permutation (NULL / 0 if not shuffled); the loader takes ownership of saved_perm when the layout matches,
// otherwise it is left to the caller to free. returns 1 if the position had to be converted.
int dataloader_resume_saved(DataLoader *loader, size_t current_shard_idx, size_t current_sample_idx,
                            int saved_num_processes, size_t saved_B, size_t saved_T,
                            int** saved_perm, size_t saved_num_samples) {
    // loads the saved shard, which sets loader->shard_num_samples for the current B, T, num_processes
    dataloader_resume(loader, current_shard_idx, current_sample_idx);
    // figure out how many tokens one saved sample was (num_processes * B * T at save time)
    size_t cur_sample_tokens = (size_t) loader->num_processes * loader->B * loader->T;
    size_t saved_sample_tokens = 0;
    if (saved_B > 0 && saved_T > 0) {
        saved_sample_tokens = (size_t) saved_num_processes * saved_B * saved_T;
    } else if (loader->should_shuffle && saved_num_samples > 0) {
        // older state file: recover B from the saved sample count of this shard, assuming T is unchanged
        // (shard_num_samples = (ntok - 1) / (num_processes * B * T), see dataloader_load_shard_)
        size_t ntok_m1 = (size_t) ((loader->file_size_bytes - (int64_t) loader->header_bytes) / sizeof(uint16_t)) - 1;
        size_t unit = (size_t) saved_num_processes * loader->T;
        for (size_t b = 1; b <= 65536; b++) {
            size_t n = ntok_m1 / (unit * b);
            if (n == saved_num_samples) { saved_sample_tokens = unit * b; saved_B = b; saved_T = loader->T; break; }
            if (n < saved_num_samples) { break; }
        }
        if (saved_sample_tokens == 0) {
            // T must have changed too; fall back to an approximate sample size
            saved_sample_tokens = ntok_m1 / saved_num_samples;
        }
    } else {
        saved_sample_tokens = cur_sample_tokens; // unknown (old file, not shuffled): assume unchanged
    }
    int changed = (saved_sample_tokens != cur_sample_tokens) ||
                  (loader->should_shuffle && saved_num_samples != loader->shard_num_samples);
    if (!changed) {
        // same layout as when saved: restore the permutation as is
        if (loader->should_shuffle) {
            free(loader->intra_shard_indices);
            loader->intra_shard_indices = *saved_perm;
            *saved_perm = NULL;
        }
        return 0;
    }
    size_t saved_sample_idx = current_sample_idx;
    dataloader_convert_position_(loader, saved_sample_idx, saved_sample_tokens, *saved_perm, saved_num_samples);
    if (loader->process_rank == 0) {
        printf("Warning: resuming with B=%zu T=%zu x%d procs (%zu tok/sample) from a state saved with B=%zu T=%zu x%d procs (%zu tok/sample): shard %zu, sample %zu/%zu -> %zu/%zu\n",
               loader->B, loader->T, loader->num_processes, cur_sample_tokens,
               saved_B, saved_T, saved_num_processes, saved_sample_tokens,
               current_shard_idx, saved_sample_idx, saved_num_samples,
               loader->current_sample_idx, loader->shard_num_samples);
    }
    return 1;
}

void dataloader_free(DataLoader *loader) {
    free(loader->buffer);
    free(loader->inputs);
    free(loader->targets);
    if (loader->should_shuffle) {
        free(loader->shard_indices);
        free(loader->intra_shard_indices);
    }
    fcloseCheck(loader->tokens_file);
    globfree(&loader->glob_result);
    free(loader->filename_pattern);
    free(loader->stream_done_path);
}

// ----------------------------------------------------------------------------
// Distributed Eval Loader
// Many evals (like) HellaSwag and MMLU are multiple-choice
// where there are 4 possible continuations and a label for the correct one
// We want to load and serve these style of evals
/*
Copy pasting the section on the eval datafile format, from data_common.py:
- First comes a header with 256 int32s
- The examples follow, each example is a stream of uint16_t:
    - <START_EXAMPLE> delimiter of 2**16-1, i.e. 65,535
    - <EXAMPLE_BYTES>, bytes encoding this example, allowing efficient skip to next
    - <EXAMPLE_INDEX>, the index of the example in the dataset
    - <LABEL>, the index of the correct completion
    - <NUM_COMPLETIONS>, indicating the number of completions (usually 4)
    - <NUM><CONTEXT_TOKENS>, where <NUM> is the number of tokens in the context
    - <NUM><COMPLETION_TOKENS>, repeated NUM_COMPLETIONS times
*/

// for now, could relax later
#define ASSUMED_NUM_COMPLETIONS 4
// helper macro for ceildiv
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

typedef struct {
    // variables related to distributed training
    // each process/worker has to access different parts of the data
    int process_rank;
    int num_processes;
    // hyperparameters. use size_t to prevent overflow
    size_t B; // (micro) batch size dimension of the tensor that feeds into the model
    size_t T; // maximum context length of the model
    // input handling and its state
    FILE* eval_file;
    uint16_t* buffer; // we fread data from file into this buffer
    // public variables that could be accessed from outside
    int num_examples; // in total across all processes
    int num_batches; // to process the entire dataset across all processes
    int start_example_index; // the assignment of work for this process, start
    int end_example_index; // and end. start is inclusive, end is exclusive
    int current_example_index; // the next example we would read
    int* inputs;  // input tokens into transformer
    int* targets; // target tokens for the transformer
    char* mask; // mask=1 at all completion token locations
    int* label; // the correct completion labels
    int num_completions; // number of completions for this example
} EvalLoader;

void evalloader_reset(EvalLoader *loader) {
    // we have to be careful that each process starts at the correct offset.
    // For example if there are N examples in the file and 4 processes,
    // then process 0 should start at 0, process 1 at N/4, process 2 at N/2, etc.
    // determine how much work there is for all processes
    int examples_per_process = CEIL_DIV(loader->num_examples, loader->num_processes);
    int can_fit_examples = (int) (loader->B / ASSUMED_NUM_COMPLETIONS);
    if (can_fit_examples == 0) {
        // this could be fixed in the future, but for now keeping it simple and throw error when B too low
        printf("HellaSwag EvalLoader: batch size %zu is < %d\n", loader->B, ASSUMED_NUM_COMPLETIONS);
        printf("---> HINT: Disable HellaSwag eval with -h 0, or increase batch size with -b\n");
        exit(EXIT_FAILURE);
    }
    loader->num_batches = CEIL_DIV(examples_per_process, can_fit_examples);
    // determine the start and end example indices for this process
    loader->start_example_index = examples_per_process * loader->process_rank;
    loader->end_example_index = examples_per_process * (loader->process_rank + 1);
    // crop the end example index to the total number of examples
    if (loader->end_example_index > loader->num_examples) {
        loader->end_example_index = loader->num_examples;
    }
    // now seek through the file to the start of that example
    // utilize <EXAMPLE_BYTES> for efficiency
    int64_t header_bytes = HEADER_SIZE * sizeof(int);
    fseekCheck(loader->eval_file, (int) header_bytes, SEEK_SET);
    for (int i = 0; i < loader->start_example_index; i++) {
        uint16_t example_header[3];
        // read 3 uint16_t values: <START_EXAMPLE>, <EXAMPLE_BYTES>, <EXAMPLE_INDEX>
        freadCheck(&example_header[0], sizeof(uint16_t), 3, loader->eval_file);
        // validate the <START_EXAMPLE> delimiter
        assert(example_header[0] == 65535); // <START_EXAMPLE> delimiter
        // validate the <EXAMPLE_INDEX>
        assert(example_header[2] == i); // <EXAMPLE_INDEX> should match the loop index
        // skip to the next example, keeping in mind that we already read the header
        size_t remaining_bytes = example_header[1] - sizeof(uint16_t) * 3;
        assert(remaining_bytes > 0); // we expect some bytes in the example
        fseekCheck(loader->eval_file, (int) remaining_bytes, SEEK_CUR);
    }
    // now we are at the start of the example we want to start at, pointing at <START_EXAMPLE>
    loader->current_example_index = loader->start_example_index;
}

void evalloader_init(EvalLoader *loader,
                     const char* filename,
                     size_t B,
                     size_t T,
                     int process_rank,
                     int num_processes) {
    loader->process_rank = process_rank;
    loader->num_processes = num_processes;
    loader->B = B;
    loader->T = T;

    // open the file and validate the header
    loader->eval_file = fopenCheck(filename, "rb");
    // validate the header
    int header[HEADER_SIZE];
    freadCheck(header, sizeof(int), HEADER_SIZE, loader->eval_file);
    if (header[0] != 20240522) { printf("Bad magic in eval file\n"); exit(EXIT_FAILURE); }
    if (header[1] != 1) { printf("Bad version in data file\n"); exit(EXIT_FAILURE); }
    loader->num_examples = header[2]; // number of examples in the file
    assert(loader->num_examples >= num_processes); // avoid headaches for now
    size_t longest_example_bytes = header[3]; // longest example in the file
    // basic sensibility check we could relax later. but roughly each example
    // contains the prompt (or "context") and 4 completions, all of these have to be
    // up to T tokens, and their tokens are uint16_t (so 2 bytes/token).
    // There's a few more things in each example but they are minor.
    // So longest example should be roughly this. Just trying to make sure it's sensible.
    assert(longest_example_bytes > 0 && longest_example_bytes < (1+ASSUMED_NUM_COMPLETIONS)*T*2);

    // allocate all the space we'll need
    int can_fit_examples = (int) (B / ASSUMED_NUM_COMPLETIONS);
    loader->buffer = (uint16_t*)mallocCheck(longest_example_bytes);
    loader->inputs = (int*)calloc(B * T, sizeof(int));
    loader->targets = (int*)calloc(B * T, sizeof(int));
    loader->mask = (char*)mallocCheck(B * T * sizeof(char));
    loader->label = (int*)mallocCheck(can_fit_examples * sizeof(int));

    // reset the loader, to initialize it
    evalloader_reset(loader);
}

void evalloader_next_example_(EvalLoader *loader, int example_batch_index) {
    // this function populates the inputs, targets, mask, and label fields for one example
    // because every (B,T) tensor can fit multiple examples and we want to take advantage,
    // we also pass in the example_batch_index to indicate which example in the batch we are loading
    // and each example takes up ASSUMED_NUM_COMPLETIONS rows in the batch
    size_t B = loader->B;
    size_t T = loader->T;
    int batch_dim_offset = example_batch_index * ASSUMED_NUM_COMPLETIONS;
    // read the current example header
    uint16_t example_header[3];
    freadCheck(&example_header[0], sizeof(uint16_t), 3, loader->eval_file);
    // validate the <START_EXAMPLE> delimiter
    assert(example_header[0] == 65535); // <START_EXAMPLE> delimiter
    // validate the <EXAMPLE_INDEX>
    assert(example_header[2] == loader->current_example_index); // <EXAMPLE_INDEX> should match the loop index
    assert(example_header[2] >= loader->start_example_index && example_header[2] < loader->end_example_index);
    // read the rest of the example (we have space for 3 more uint16_t values in buffer, it's ok)
    size_t example_bytes = example_header[1] - sizeof(uint16_t) * 3;
    // read example_bytes into buffer. careful that this is actually in the units of bytes
    freadCheck(loader->buffer, sizeof(char), example_bytes, loader->eval_file);
    // process the example label
    int label = (int)loader->buffer[0];
    int can_fit_examples = (int) (loader->B / ASSUMED_NUM_COMPLETIONS);
    assert(label >= 0 && label < ASSUMED_NUM_COMPLETIONS); // we expect the label to be in [0, 4) for right now
    assert(example_batch_index >= 0 && example_batch_index < can_fit_examples);
    loader->label[example_batch_index] = label; // store for output
    // process the number of completions
    int num_completions = (int)loader->buffer[1];
    assert(num_completions == ASSUMED_NUM_COMPLETIONS); // we expect 4 completions for now
    assert(batch_dim_offset + num_completions <= B); // we expect to fit in the batch
    loader->num_completions = num_completions; // store for output
    // process the context
    // the context is shared for all completions, so we insert it into all data rows equally
    int context_length = (int)loader->buffer[2];
    uint16_t *context_tokens_start = &loader->buffer[3]; // where the tokens start
    assert(context_length > 0 && context_length < T); // context is non-empty and up to T
    for (int b = 0; b < num_completions; b++) {
        for (int i = 0; i < context_length; i++) {
            int boff = batch_dim_offset + b;
            int tok_cur = (int)context_tokens_start[i];
            loader->inputs[boff * T + i] = tok_cur;
        }
    }
    // process the completions, insert them in their row, right after the (shared) context
    uint16_t *completions_iter = loader->buffer + 3 + context_length;
    for (int c = 0; c < num_completions; c++) {
        int coff = batch_dim_offset + c;
        int completion_length = (int)completions_iter[0];
        uint16_t *completion_tokens_start = completions_iter + 1;
        assert(completion_length > 0 && context_length + completion_length < T); // things fit?
        for (int i = 0; i < completion_length; i++) {
            int tok_cur = (int)completion_tokens_start[i];
            // at inputs, the completions simply follow the context
            loader->inputs[coff * T + context_length + i] = tok_cur;
            // at targets things start to get tricky
            // we expect the last context token to predict the first completion token
            // and then onwards from there.
            loader->targets[coff * T + context_length + i - 1] = tok_cur;
            // and at these positions, we want to set mask=1, because these are the
            // positions where we want to average the loss, in each row, to determine
            // its overall probability of following the context.
            loader->mask[coff * T + context_length + i - 1] = 1;
        }
        completions_iter += 1 + completion_length; // move to the next completion
    }
    // advance the current example to point to the next one we'd load
    loader->current_example_index += 1;
}

void evalloader_next_batch(EvalLoader *loader) {
    size_t B = loader->B;
    size_t T = loader->T;
    // init mask to zeros, no need to do it for inputs & targets, the values where the mask
    // is set will be correctly overwritten every time.
    memset(loader->mask, 0, B * T * sizeof(char));
    // ok here is the problem we are solving
    // we have a batch dimension of B, which we want to take full advantage of
    // each example has some number of completions (usually 4)
    // so we want to pack as many examples into rows of B as we can fit
    int can_fit_examples = (int) (B / ASSUMED_NUM_COMPLETIONS); // how many examples can we fit in the batch?
    for (int i = 0; i < can_fit_examples; i++) {
        if (loader->current_example_index >= loader->end_example_index) {
            break; // this process has exhausted its work, noop from here on
        }
        evalloader_next_example_(loader, i);
    }
}

int evalloader_stat_losses(EvalLoader *loader, float* losses) {
    // compute statistics of losses (B*T) resulting from a forward pass
    // on a batch that was constructed from EvalLoader
    // putting this functionality here because it is tightly coupled
    // with how we construct and represent the data batches.
    // returns the number of correct examples in this batch.
    int correct = 0;
    size_t B = loader->B;
    size_t T = loader->T;
    // iterate the examples in this batch
    int can_fit_examples = (int) (B / ASSUMED_NUM_COMPLETIONS);
    for (int i = 0; i < can_fit_examples; i++) {
        float min_loss = 0.0f;
        int min_loss_index = -1;
        char active = 0; // is this example active or fully empty?
        // iterate the completions in this example
        for (int b = 0; b < ASSUMED_NUM_COMPLETIONS; b++) {
            int boff = i * ASSUMED_NUM_COMPLETIONS + b;
            // evaluate the quality of this completion
            // its quality is simply the average loss over the tokens
            float average_loss = 0.0f;
            int count = 0;
            for (int t = 0; t < T; t++) {
                char mask = loader->mask[boff * T + t];
                if (mask == 1) {
                    active = 1;
                    average_loss += losses[boff * T + t];
                    count++;
                }
            }
            if (count > 0) { average_loss /= count; }
            if (b == 0 || average_loss < min_loss) {
                min_loss = average_loss;
                min_loss_index = b;
            }
        }
        if (active && (min_loss_index == loader->label[i])) {
            correct += 1;
        }
    }
    return correct;
}

void evalloader_free(EvalLoader *loader) {
    free(loader->buffer);
    free(loader->inputs);
    free(loader->targets);
    free(loader->mask);
    free(loader->label);
    fcloseCheck(loader->eval_file);
}

#endif // DATALOADER_H