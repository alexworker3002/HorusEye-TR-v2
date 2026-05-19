#!/usr/bin/env bash
# 中文说明：本脚本创建于 2026-05-19，用于生成新的 HorusEye-TR 实验矩阵脚本。
# 编写情景：后续需要新增实验矩阵时，不直接手写空白 .sh，而是通过本脚本写入标准中文头部。
# 测试用途：保证每个矩阵脚本开头都记录实验背景、测试目的、预期结果、创建时间和输出位置。
# 预期结果：新建矩阵文件具备统一可读的自描述头部，降低多轮训练实验之间的记录混乱。
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "Usage: $0 <target-script> <created-at> <context> <test-purpose> <expected-result>" >&2
  exit 2
fi

target="$1"
created_at="$2"
context="$3"
purpose="$4"
expected="$5"

if [[ -e "$target" ]]; then
  echo "Refusing to overwrite existing script: $target" >&2
  exit 1
fi

mkdir -p "$(dirname "$target")"
cat > "$target" <<EOF
#!/usr/bin/env bash
# 中文说明：本实验矩阵脚本创建于 ${created_at}。
# 编写情景：${context}
# 测试用途：${purpose}
# 预期结果：${expected}
set -euo pipefail

cd /home/ice/workspace/HorusEye-TR-v2

# TODO: 在这里填写训练矩阵、评估矩阵和汇总逻辑。
EOF
chmod +x "$target"
echo "Created experiment matrix script: $target"
