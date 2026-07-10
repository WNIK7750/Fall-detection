"""
Git 操作脚本：初始化仓库 → 提交代码 → 推送到 GitHub
使用系统 Git (C:\\Program Files\\Git\\cmd\\git.exe)
"""
import subprocess
import os
import sys

PROJECT_DIR = r"D:\识别人类摔倒的智能系统"
GIT_EXE = r"C:\Program Files\Git\cmd\git.exe"
REMOTE_URL = "https://github.com/WNIK7750/Fall-detection.git"

def run_git(*args, cwd=PROJECT_DIR, check=True):
    """执行 git 命令并打印输出"""
    cmd = [GIT_EXE] + list(args)
    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.stderr.strip():
        print(f"[stderr] {result.stderr.strip()}")
    if check and result.returncode != 0:
        print(f"[ERROR] git {' '.join(args)} failed with code {result.returncode}")
        sys.exit(1)
    return result

def main():
    os.chdir(PROJECT_DIR)

    # 1. 初始化 Git 仓库
    if not os.path.isdir(os.path.join(PROJECT_DIR, ".git")):
        run_git("init")

    # 2. 设置默认分支为 main
    run_git("checkout", "-b", "main", check=False)

    # 3. 配置用户信息（如果还没配置的话）
    # 检查是否已有 user.name
    name_result = run_git("config", "user.name", check=False)
    if not name_result.stdout.strip():
        run_git("config", "user.name", "偶散成灰")
    email_result = run_git("config", "user.email", check=False)
    if not email_result.stdout.strip():
        run_git("config", "user.email", "WNIK7750@users.noreply.github.com")

    # 4. 添加远程仓库
    run_git("remote", "remove", "origin", check=False)
    run_git("remote", "add", "origin", REMOTE_URL)

    # 5. 添加所有文件（遵循 .gitignore）
    run_git("add", "-A")

    # 6. 查看暂存区状态
    status = run_git("status", "--short")
    file_count = len([l for l in status.stdout.strip().split('\n') if l.strip()])
    print(f"\n=== 暂存文件数: {file_count} ===")

    # 7. 提交
    commit_msg = "Initial commit: 智能摔倒检测系统

- 基于 YOLOv8-Pose 的姿态估计 + LSTM 时序分析的摔倒检测
- 包含训练、测试、集成推理、特征提取等完整代码
- .gitignore 排除了数据集视频、模型权重、虚拟环境等大文件"
    run_git("commit", "-m", commit_msg, check=False)

    # 8. 推送到 GitHub
    print("\n=== 推送到 GitHub ===")
    push_result = run_git("push", "-u", "origin", "main", check=False)

    if push_result.returncode != 0:
        # 可能分支名不是 main，尝试 master
        print("\n[RETRY] 尝试推送 master 分支...")
        run_git("push", "-u", "origin", "master", check=False)

    print("\n=== 完成! ===")
    run_git("status")

if __name__ == "__main__":
    main()
