#include <metal_stdlib>
using namespace metal;

kernel void selective_scan_stub(
    device const float* gates   [[buffer(0)]],
    device const float* updates [[buffer(1)]],
    device float* states        [[buffer(2)]],
    constant uint& hidden_dim   [[buffer(3)]],
    constant uint& seq_len      [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid == 0) {
        states[0] = 0.0f;
    }
}
