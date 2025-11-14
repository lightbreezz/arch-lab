#!/bin/bash

# 源文件
SRC="matrix_mul.cu"
BACKUP="${SRC}.bak"

# 如果没有备份，先创建一个原始备份（只做一次）
if [ ! -f "$BACKUP" ]; then
    cp "$SRC" "$BACKUP"
    echo "原始文件已备份为 $BACKUP"
fi

# 要测试的 TILE_WIDTH 值列表
tile_widths=(8 16 32)

NVCC_FLAGS="-L/usr/local/cuda/lib64 -lcublas"

# 结果日志文件
LOG="tile_performance.log"
echo "TILE_WIDTH, GFLOP/s, Time (ms)" > "$LOG"

for tw in "${tile_widths[@]}"; do
    echo "=============================="
    echo "Testing TILE_WIDTH = $tw"
    echo "=============================="

    # 恢复原始文件
    cp "$BACKUP" "$SRC"

    # 使用 sed 替换 TILE_WIDTH 的值（只替换定义行）
    sed -i "s/const int TILE_WIDTH = [0-9]\+;/const int TILE_WIDTH = $tw;/" "$SRC"

    # 编译
    echo "Compiling with TILE_WIDTH=$tw..."
    nvcc $NVCC_FLAGS "$SRC" -o a.out
    if [ $? -ne 0 ]; then
        echo "编译失败，跳过 TILE_WIDTH=$tw"
        echo "$tw, COMPILE ERROR, -" >> "$LOG"
        continue
    fi

    # 运行
    echo "Running ./a.out 0 4000 ..."
    output=$(./a.out 0 4000 2>&1)

    # 提取 GFLOP/s 和 Time
    gflops=$(echo "$output" | grep "Performance=" | grep -oP 'Performance=\s*\K[0-9.]+')
    time_ms=$(echo "$output" | grep "Performance=" | grep -oP 'Time=\s*\K[0-9.]+')

    if [ -z "$gflops" ]; then
        gflops="RUN ERROR"
        time_ms="-"
    fi

    echo "Result: GFLOP/s = $gflops, Time = ${time_ms}ms"
    echo "$tw, $gflops, $time_ms" >> "$LOG"
done

echo "测试完成，结果保存在 $LOG"