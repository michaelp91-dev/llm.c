/*
Tests the streaming mode of the DataLoader (dataloader_init_streaming), where shards
are still being written by a producer while training reads them.

compile and run as (from dev/test directory)
gcc -O3 -I../../llmc -o test_dataloader_stream test_dataloader_stream.c -lm -lpthread && ./test_dataloader_stream
*/
#include <unistd.h>
#include <pthread.h>
#include <sys/stat.h>
#include "../../llmc/dataloader.h"

#define DIR "stream_test_shards"
const int num_tokens = 140; // tokens per shard
const int num_shards = 4;
int B = 4;
int T = 8;

void write_shard(int i, int delay_ms) {
    // tokens of shard i are i*num_tokens ... (i+1)*num_tokens-1, written to a temp name then renamed
    if (delay_ms > 0) { usleep(delay_ms * 1000); }
    char tmp[256], name[256];
    snprintf(tmp, sizeof(tmp), DIR "/shard_%04d.bin.tmp", i);
    snprintf(name, sizeof(name), DIR "/shard_%04d.bin", i);
    FILE* f = fopenCheck(tmp, "wb");
    int header[HEADER_SIZE] = {0};
    header[0] = 20240520; header[1] = 1; header[2] = num_tokens;
    fwrite(header, sizeof(int), HEADER_SIZE, f);
    for (int t = 0; t < num_tokens; t++) { uint16_t tok = (uint16_t)(i * num_tokens + t); fwrite(&tok, 2, 1, f); }
    fcloseCheck(f);
    rename(tmp, name);
}

void clean(void) {
    char name[256];
    for (int i = 0; i < num_shards; i++) { snprintf(name, sizeof(name), DIR "/shard_%04d.bin", i); remove(name); }
    remove(DIR "/STREAM_DONE");
    mkdir(DIR, 0755);
}

typedef struct { int delay_ms; int write_done; } ProducerArgs;
void* producer(void* arg) {
    ProducerArgs* a = (ProducerArgs*)arg;
    for (int i = 1; i < num_shards; i++) { write_shard(i, a->delay_ms); }
    if (a->write_done) { FILE* f = fopenCheck(DIR "/STREAM_DONE", "w"); fprintf(f, "%d\n", num_shards); fcloseCheck(f); }
    return NULL;
}

void check(int ok, const char* msg) { if (!ok) { printf("FAIL: %s\n", msg); exit(EXIT_FAILURE); } }

int first_token_of_batch(DataLoader* l) { return l->inputs[0]; }

void test_stream_in_order(int write_done, int expected_shards) {
    // only shard 0 exists at init; others arrive slowly. tokens must come in order, never wrapping early
    printf("test_stream_in_order(done_marker=%d, expected=%d)... ", write_done, expected_shards); fflush(stdout);
    clean();
    write_shard(0, 0);
    DataLoader loader;
    dataloader_init_streaming(&loader, DIR "/shard_*.bin", B, T, 0, 1, 0, expected_shards);
    pthread_t th; ProducerArgs a = {300, write_done};
    pthread_create(&th, NULL, producer, &a);
    int per_shard = (num_tokens - 1) / (B * T); // 4 batches per shard
    for (int s = 0; s < num_shards; s++) {
        for (int k = 0; k < per_shard; k++) {
            dataloader_next_batch(&loader);
            for (int i = 0; i < B * T; i++) {
                check(loader.inputs[i] == s * num_tokens + k * B * T + i, "input token mismatch");
                check(loader.targets[i] == s * num_tokens + k * B * T + i + 1, "target token mismatch");
            }
        }
    }
    pthread_join(th, NULL);
    // producer is done: the next batch wraps to shard 0 (a new epoch)
    dataloader_next_batch(&loader);
    check(loader.inputs[0] == 0, "expected wrap to shard 0 after producer finished");
    check(loader.glob_result.gl_pathc == (size_t)num_shards, "expected all shards to be known");
    dataloader_free(&loader);
    printf("OK\n");
}

void test_stream_shuffled_multiprocess(void) {
    // shuffle within shards, 2 processes. each shard must be fully consumed (disjointly across ranks)
    // before the next one, and shards are visited in filename order
    printf("test_stream_shuffled_multiprocess... "); fflush(stdout);
    clean();
    write_shard(0, 0);
    DataLoader l0, l1;
    dataloader_init_streaming(&l0, DIR "/shard_*.bin", B, T, 0, 2, 1, 0);
    dataloader_init_streaming(&l1, DIR "/shard_*.bin", B, T, 1, 2, 1, 0);
    pthread_t th; ProducerArgs a = {200, 1};
    pthread_create(&th, NULL, producer, &a);
    int per_shard = (num_tokens - 1) / (2 * B * T); // 2 global batches per shard
    for (int s = 0; s < num_shards; s++) {
        int seen[140] = {0};
        for (int k = 0; k < per_shard; k++) {
            dataloader_next_batch(&l0);
            dataloader_next_batch(&l1);
            check(l0.current_shard_idx == (size_t)s && l1.current_shard_idx == (size_t)s, "ranks disagree on shard");
            for (int i = 0; i < B * T; i++) {
                int t0 = l0.inputs[i] - s * num_tokens, t1 = l1.inputs[i] - s * num_tokens;
                check(t0 >= 0 && t0 < num_tokens && t1 >= 0 && t1 < num_tokens, "token from wrong shard");
                seen[t0]++; seen[t1]++;
            }
        }
        for (int i = 0; i < per_shard * 2 * B * T; i++) { check(seen[i] == 1, "token not seen exactly once"); }
    }
    pthread_join(th, NULL);
    dataloader_free(&l0);
    dataloader_free(&l1);
    printf("OK\n");
}

void test_stream_resume_waits(void) {
    // resuming at a shard that does not exist yet (e.g. data being regenerated after a disconnect) must wait for it
    printf("test_stream_resume_waits... "); fflush(stdout);
    clean();
    write_shard(0, 0);
    DataLoader loader;
    dataloader_init_streaming(&loader, DIR "/shard_*.bin", B, T, 0, 1, 0, num_shards);
    pthread_t th; ProducerArgs a = {300, 0}; // no marker: done is signalled by expected_shards
    pthread_create(&th, NULL, producer, &a);
    dataloader_resume(&loader, 2, 1);
    dataloader_next_batch(&loader);
    check(loader.inputs[0] == 2 * num_tokens + 1 * B * T, "resume position mismatch");
    pthread_join(th, NULL);
    dataloader_free(&loader);
    printf("OK\n");
}

void test_non_stream_unchanged(void) {
    // the default loader with the same files must wrap normally and not wait
    printf("test_non_stream_unchanged... "); fflush(stdout);
    clean();
    write_shard(0, 0); write_shard(1, 0);
    DataLoader loader;
    dataloader_init(&loader, DIR "/shard_*.bin", B, T, 0, 1, 0);
    check(loader.stream == 0, "stream flag set");
    for (int k = 0; k < 8; k++) dataloader_next_batch(&loader);
    dataloader_next_batch(&loader);
    check(loader.inputs[0] == 0, "expected wrap to shard 0");
    dataloader_free(&loader);
    printf("OK\n");
}

int main(void) {
    test_non_stream_unchanged();
    test_stream_in_order(1, 0);
    test_stream_in_order(0, num_shards);
    test_stream_shuffled_multiprocess();
    test_stream_resume_waits();
    clean();
    rmdir(DIR);
    printf("all streaming dataloader tests passed\n");
    return 0;
}
