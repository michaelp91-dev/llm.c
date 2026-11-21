/*
Tests resuming the DataLoader from a saved state when the micro batch size B (or the number of
processes) differs from the run that saved it (dataloader_resume_saved in llmc/dataloader.h).
Mirrors what load_state() in train_gpt2.cu does with the dataloader part of a state file.

compile and run as (from dev/test directory)
gcc -O2 -I../../llmc -o test_dataloader_resume test_dataloader_resume.c -lm && ./test_dataloader_resume
*/
#include <sys/stat.h>
#include "../../llmc/dataloader.h"

#define DIR "resume_test_shards"
#define NTOK 60001   // tokens per shard; token value == position in shard, so we can track coverage
#define NSHARDS 3
const size_t T = 16;

void write_shards(void) {
    mkdir(DIR, 0755);
    for (int i = 0; i < NSHARDS; i++) {
        char name[256];
        snprintf(name, sizeof(name), DIR "/shard_%04d.bin", i);
        FILE* f = fopenCheck(name, "wb");
        int header[HEADER_SIZE] = {0};
        header[0] = 20240520; header[1] = 1; header[2] = NTOK;
        fwrite(header, sizeof(int), HEADER_SIZE, f);
        for (int t = 0; t < NTOK; t++) { uint16_t tok = (uint16_t) t; fwrite(&tok, 2, 1, f); }
        fcloseCheck(f);
    }
    FILE* f = fopenCheck(DIR "/STREAM_DONE", "w"); fcloseCheck(f);
}

int counts[NTOK];
// count tokens of one batch (all processes: we emulate num_processes by reading B*T*procs per sample)
void count_batch(DataLoader* l) {
    for (size_t i = 0; i < l->B * l->T; i++) { counts[l->inputs[i]]++; }
}

int failures = 0;
#define CHECK(c, ...) do { if (!(c)) { printf("FAIL: " __VA_ARGS__); printf("\n"); failures++; } } while (0)

// run saved_B for `steps` samples (crossing into shard 1), "save", resume at new_B, finish shard 1
void run_case(size_t saved_B, size_t new_B, int stream, int shuffle, size_t steps, int header_has_BT) {
    const char* pattern = DIR "/shard_*.bin";
    DataLoader a;
    if (stream) dataloader_init_streaming(&a, pattern, saved_B, T, 0, 1, shuffle, NSHARDS);
    else dataloader_init(&a, pattern, saved_B, T, 0, 1, shuffle);
    memset(counts, 0, sizeof(counts));
    for (size_t s = 0; s < steps; s++) {
        dataloader_next_batch(&a);
        if (a.current_shard_idx == 1) { count_batch(&a); }
    }
    assert(a.current_shard_idx == 1);
    // --- what save_state writes ---
    size_t sv_shard = a.current_shard_idx, sv_sample = a.current_sample_idx, sv_n = a.shard_num_samples;
    int* sv_perm = NULL;
    if (shuffle) {
        sv_perm = (int*) mallocCheck(sv_n * sizeof(int));
        memcpy(sv_perm, a.intra_shard_indices, sv_n * sizeof(int));
    }
    mt19937_state sv_rng = a.shuffle_rng;
    long consumed_before = 0;
    for (int t = 0; t < NTOK; t++) consumed_before += counts[t];
    dataloader_free(&a);

    // --- what load_state does on the new run ---
    DataLoader b;
    if (stream) dataloader_init_streaming(&b, pattern, new_B, T, 0, 1, shuffle, NSHARDS);
    else dataloader_init(&b, pattern, new_B, T, 0, 1, shuffle);
    b.should_shuffle = shuffle;
    if (shuffle) b.shuffle_rng = sv_rng;
    int conv = dataloader_resume_saved(&b, sv_shard, sv_sample, 1,
                                       header_has_BT ? saved_B : 0, header_has_BT ? T : 0, &sv_perm, sv_n);
    free(sv_perm);
    CHECK(conv == (saved_B != new_B), "converted=%d for B %zu->%zu", conv, saved_B, new_B);
    CHECK(b.current_shard_idx == 1, "resumed in shard %zu", b.current_shard_idx);
    // finish shard 1
    long served_after = 0;
    while (1) {
        dataloader_next_batch(&b);
        if (b.current_shard_idx != 1) break;
        count_batch(&b);
        served_after += (long)(new_B * T);
    }
    // coverage of shard 1
    long zero = 0, twice = 0, more = 0;
    size_t usable = (size_t)((NTOK - 1) / (new_B * T)) * new_B * T; // tokens the new layout can serve
    for (size_t t = 0; t < usable; t++) {
        if (counts[t] == 0) zero++; else if (counts[t] == 2) twice++; else if (counts[t] > 2) more++;
    }
    long total = consumed_before + served_after;
    printf("B %2zu->%2zu stream=%d shuffle=%d hdrBT=%d: before %6ld tok, after %6ld tok, shard usable %6zu | skipped %5ld repeated %5ld (>2x: %ld)\n",
           saved_B, new_B, stream, shuffle, header_has_BT, consumed_before, served_after, usable, zero, twice, more);
    CHECK(more == 0, "tokens served more than twice");
    if (saved_B == new_B) {
        CHECK(zero == 0 && twice == 0, "same B must resume exactly");
    } else {
        // only straddling samples may be skipped / repeated: at most one saved sample's worth each, per new sample
        long bound = (long)((usable / (new_B * T) + 1) * (saved_B > new_B ? saved_B : new_B) * T);
        CHECK(zero <= bound && twice <= bound, "too much skipped/repeated");
        // and the total stays close to one pass over the shard
        long diff = total - (long) usable; if (diff < 0) diff = -diff;
        CHECK(diff <= (long)(usable / 10 + 2 * saved_B * T + 2 * new_B * T), "total %ld vs usable %zu", total, usable);
    }
    dataloader_free(&b);
}

int main(void) {
    write_shards();
    size_t per_shard_8 = (NTOK - 1) / (8 * T);
    // the real case: state saved at B=8 (old file, no B/T in header), resumed at B=16, streaming + shuffled
    for (int st = 0; st <= 1; st++) {
        for (int hdr = 0; hdr <= 1; hdr++) {
            run_case(8, 16, st, 1, per_shard_8 + 37, hdr);
            run_case(8, 16, st, 1, per_shard_8 + per_shard_8 / 2, hdr);
            run_case(16, 8, st, 1, (NTOK - 1) / (16 * T) + 50, hdr);
            run_case(8, 12, st, 1, per_shard_8 + 101, hdr);
            run_case(8, 8, st, 1, per_shard_8 + 37, hdr);
        }
        run_case(8, 16, st, 0, per_shard_8 + 37, 1);
        run_case(8, 8, st, 0, per_shard_8 + 37, 0);
    }
    // edge: saved exactly at the end of shard 1's samples is handled by next_batch advancing
    printf(failures ? "FAILED: %d\n" : "all resume tests passed\n", failures);
    return failures != 0;
}
