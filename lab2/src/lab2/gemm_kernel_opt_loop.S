.text;
.p2align 2;
.global gemm_kernel_opt_loop;
.type gemm_kernel_opt_loop, %function;

#define     MAT_C               %rdi
#define     MAT_A               %rsi
#define     MAT_B               %r14
#define     DIM_M               %rcx
#define     DIM_N               %r8
#define     DIM_K               %r9
#define     loop_m              %r10
#define     loop_k              %r11
#define     loop_n              %r12
#define     mat_elem_idx        %r13
; #define     prefetch_elem_idx   %r15
// 预取寄存器暂不使用
// 注意rbx rdx 使用注意
#define     block_m             %rbx
#define     block_n             %rdx
#define     BLOCK_SIZE          $16     

.macro PUSHD   // 保存原通用寄存器值
    push %rax
    push %rbx
    push %rcx
    push %rdx
    push %rsi
    push %rdi
    push %rbp
    push %r8
    push %r9
    push %r10
    push %r11
    push %r12
    push %r13
    push %r14
    push %r15
.endm

.macro POPD    // 恢复原通用寄存器值
    pop %r15
    pop %r14
    pop %r13
    pop %r12
    pop %r11
    pop %r10
    pop %r9
    pop %r8
    pop %rbp
    pop %rdi
    pop %rsi
    pop %rdx
    pop %rcx
    pop %rbx
    pop %rax
.endm

.macro GEMM_INIT
    mov %rdx, MAT_B
    xor loop_m, loop_m
    xor loop_k, loop_k
    xor loop_n, loop_n
    xor block_m, block_m
    xor block_n, block_n
.endm

.macro DO_GEMM

DO_BLOCK_M:
    xor block_n, block_n

DO_BLOCK_N:
    xor loop_k, loop_k

DO_LOOP_K:
    mov block_m, loop_m
    // 计算当前 m 分块的结束行：min(block_m + 16, M)
    mov block_m, %rax
    add BLOCK_SIZE, %rax       // %rax = block_m + 16
    cmp DIM_M, %rax
    cmovg DIM_M, %rax          // if %rax > M, then %rax = M
    mov %rax, %r15             // %r15 = m_end

DO_LOOP_M:
    mov block_n, loop_n
    mov block_n, %rax
    add BLOCK_SIZE, %rax
    cmp DIM_N, %rax
    cmovg DIM_N, %rax
    mov %rax, %rbp             // %rbp = n_end

    // A[m][k]
    mov loop_m, %rax
    imul DIM_K, %rax
    add loop_k, %rax
    flds (MAT_A, %rax, 4)

DO_LOOP_N:
    mov loop_n, %rax
    and $7, %rax              // 每8个元素预取一次, 8*4=32字节缓存行
    jnz NO_PREFETCH

    // A[m+16][k]
    mov loop_m, %rax
    add $16, %rax
    cmp %r15, %rax                  
    jge NO_PREFETCH

    imul DIM_K, %rax
    add loop_k, %rax
    prefetchnta (MAT_A, %rax, 4)

    // B[k][n+16]
    mov loop_n, %rax
    add $16, %rax
    cmp %rbp, %rax
    jge NO_PREFETCH

    mov loop_k, %rax
    imul DIM_N, %rax
    add loop_n, %rax
    add $16, %rax
    prefetcht1 (MAT_B, %rax, 4)

    // C[m][n+16]
    mov loop_m, %rax
    imul DIM_N, %rax
    add loop_n, %rax
    add $16, %rax
    prefetcht1 (MAT_C, %rax, 4)
NO_PREFETCH:
    // 正常计算
    mov DIM_N, %rax
    mul loop_k
    mov %rax, mat_elem_idx
    add loop_n, mat_elem_idx                    // 计算 k*N+n
    flds (MAT_B, mat_elem_idx, 4)     // 加载 B[k][n]

    fmul %st(1), %st(0)                 // 计算A[m][k] * B[k][n]

    mov DIM_N, %rax
    mul loop_m
    mov %rax, mat_elem_idx
    add loop_n, mat_elem_idx                    // 计算 m*N+n
    flds (MAT_C, mat_elem_idx, 4)     // 加载 C[m][n]

    faddp %st(1), %st(0)                 // 计算 C[m][n] + A[m][k] * B[k][n]
    fstps (MAT_C, mat_elem_idx, 4)

    add $1, loop_n
    cmp %rbp, loop_n
    jl DO_LOOP_N

    fstp %st(0)                   // 仅弹出元素
    add $1, loop_m
    cmp %r15, loop_m
    jl DO_LOOP_M

    add $1, loop_k
    cmp DIM_K, loop_k
    jl DO_LOOP_K

    add BLOCK_SIZE, block_n
    cmp DIM_N, block_n
    jl DO_BLOCK_N

    add BLOCK_SIZE, block_m
    cmp DIM_M, block_m
    jl DO_BLOCK_M
.endm

gemm_kernel_opt_loop:
    PUSHD
    GEMM_INIT
    DO_GEMM
    POPD
    ret
    


