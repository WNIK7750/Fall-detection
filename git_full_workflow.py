"""
Git 全流程操作脚本：
  1. 初始化 Git 仓库
  2. 配置用户信息
  3. 添加远程仓库
  4. 提交代码（遵循 .gitignore，排除大文件）
  5. 推送到 GitHub
  6. 邀请协作者（需要 gh CLI 或 GitHub Token）

使用方法：
  在项目根目录打开终端（PowerShell 或 CMD），执行：
    python git_full_workflow.py

  如果需要邀请协作者，在脚本底部修改 COLLABORATORS 列表。
"""
import subprocess
import os
import sys

PROJECT_DIR = r"D:\识别人类摔倒的智能系统"
GIT_EXE = r"C:\Program Files\Git\cmd\git.exe"
REMOTE_URL = "https://github.com/WNIK7750/Fall-detection.git"
REPO_OWNER = "WNIK7750"
REPO_NAME = "Fall-detection"

# ====== 协作者 GitHub 用户名（改成你实际要邀请的人）======
COLLABORATORS = [
    # "username1",
    # "username2",
]

def run(cmd, cwd=PROJECT_DIR, check=True, label=""):
    """执行命令并打印输出"""
    print(f"\n{'='*60}")
    print(f"[{label}] {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                           encoding='utf-8', errors='replace')
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.stderr.strip():
        print(f"[stderr] {result.stderr.strip()}")
    if check and result.returncode != 0:
        print(f"[ERROR] 命令失败，退出码 {result.returncode}")
        sys.exit(1)
    return result

def run_git(*args, check=True):
    return run([GIT_EXE] + list(args), check=check, label=f"git {' '.join(args[:2])}")

def main():
    os.chdir(PROJECT_DIR)
    print(f"工作目录: {os.getcwd()}")

    # ===== 1. 初始化 Git 仓库 =====
    if not os.path.isdir(os.path.join(PROJECT_DIR, ".git")):
        run_git("init")
    else:
        print("\n[INFO] Git 仓库已存在，跳过 init")

    # ===== 2. 设置分支为 main =====
    # 检查当前分支
    branch_result = run_git("symbolic-ref", "--short", "HEAD", check=False)
    current_branch = branch_result.stdout.strip()
    if current_branch and current_branch != "main":
        run_git("branch", "-M", "main")
    elif not current_branch:
        # 还没有任何 commit，直接创建 main 分支
        run_git("symbolic-ref", "HEAD", "refs/heads/main")

    # ===== 3. 配置用户信息（仅当未配置时）======
    name_result = run_git("config", "user.name", check=False)
    if not name_result.stdout.strip():
        run_git("config", "user.name", "偶散成灰")

    email_result = run_git("config", "user.email", check=False)
    if not email_result.stdout.strip():
        run_git("config", "user.email", "WNIK7750@users.noreply.github.com")

    print(f"\n[INFO] Git 用户: {run_git('config', 'user.name').stdout.strip()}")
    print(f"[INFO] Git 邮箱: {run_git('config', 'user.email').stdout.strip()}")

    # ===== 4. 添加远程仓库 =====
    run_git("remote", "remove", "origin", check=False)
    run_git("remote", "add", "origin", REMOTE_URL)
    print(f"[INFO] 远程仓库: {REMOTE_URL}")

    # ===== 5. 添加文件（.gitignore 会自动排除大文件）======
    run_git("add", "-A")

    # ===== 6. 查看暂存区状态 =====
    status = run_git("status", "--short")
    files = [l for l in status.stdout.strip().split('\n') if l.strip()]
    print(f"\n[INFO] 暂存文件数: {len(files)}")
    # 打印前 30 个文件
    for f in files[:30]:
        print(f"  {f}")
    if len(files) > 30:
        print(f"  ... 还有 {len(files) - 30} 个文件")

    # ===== 7. 提交 =====
    commit_msg = (
        "Initial commit: 智能摔倒检测系统\n\n"
        "- 基于 YOLOv8-Pose 的姿态估计 + LSTM 时序分析的摔倒检测\n"
        "- 包含训练、测试、集成推理、特征提取等完整代码\n"
        "- .gitignore 排除了数据集视频、模型权重、虚拟环境等大文件"
    )
    commit_result = run_git("commit", "-m", commit_msg, check=False)
    if commit_result.returncode == 0:
        print("[INFO] 提交成功!")
    else:
        print("[INFO] 可能没有新变更需要提交，或提交失败")

    # ===== 8. 推送到 GitHub =====
    print("\n" + "=" * 60)
    print("正在推送到 GitHub...")
    print("=" * 60)
    push_result = run_git("push", "-u", "origin", "main", check=False)

    if push_result.returncode != 0:
        print("\n[RETRY] main 分支推送失败，尝试 master...")
        run_git("push", "-u", "origin", "master", check=False)

    # ===== 9. 邀请协作者 =====
    if COLLABORATORS:
        print("\n" + "=" * 60)
        print("邀请协作者")
        print("=" * 60)
        invite_collaborators()
    else:
        print("\n[INFO] 没有配置协作者。")
        print("       要邀请协作者，请编辑本脚本顶部的 COLLABORATORS 列表，")
        print("       填入两位协作者的 GitHub 用户名后重新运行。")
        print("       或者稍后在 GitHub 仓库页面手动邀请。")

    # ===== 最终状态 =====
    print("\n" + "=" * 60)
    print("最终状态")
    print("=" * 60)
    run_git("status")
    run_git("log", "--oneline", "-5", check=False)

    print("\n" + "=" * 60)
    print("全部完成!")
    print(f"仓库地址: {REMOTE_URL}")
    print("=" * 60)

def invite_collaborators():
    """通过 gh CLI 邀请协作者"""
    # 检查 gh CLI 是否可用
    gh_paths = [
        r"C:\Program Files\GitHub CLI\gh.exe",
        r"C:\Program Files (x86)\GitHub CLI\gh.exe",
        "gh",  # 如果在 PATH 中
    ]
    gh_exe = None
    for p in gh_paths:
        try:
            result = subprocess.run([p, "--version"], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                gh_exe = p
                break
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    if not gh_exe:
        print("[WARN] 未找到 GitHub CLI (gh)，无法自动邀请协作者。")
        print("       请在 GitHub 仓库页面手动邀请：")
        print(f"       {REMOTE_URL}/settings/access")
        return

    for username in COLLABORATORS:
        print(f"\n邀请 {username} 为协作者 (admin 权限)...")
        result = subprocess.run(
            [gh_exe, "api",
             f"/repos/{REPO_OWNER}/{REPO_NAME}/collaborators/{username}",
             "-X", "PUT",
             "-f", "permission=admin"],
            capture_output=True, text=True, encoding='utf-8', errors='replace'
        )
        if result.returncode == 0:
            print(f"[OK] 已发送邀请给 {username}")
        else:
            print(f"[ERROR] 邀请 {username} 失败: {result.stderr}")

if __name__ == "__main__":
    main()
