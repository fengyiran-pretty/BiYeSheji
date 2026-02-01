#!/bin/bash
# WSL项目自动安装脚本
# 用法: bash wsl_setup.sh

set -e  # 遇到错误立即退出

echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                          ║"
echo "║           BiYeSheji 项目 WSL 自动安装脚本                                ║"
echo "║                                                                          ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 项目信息
PROJECT_NAME="BiYeSheji"
GITHUB_URL="https://github.com/fengyiran-pretty/BiYeSheji.git"
INSTALL_DIR="$HOME/projects"
PROJECT_DIR="$INSTALL_DIR/$PROJECT_NAME"

# 打印彩色消息
print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# 检查命令是否存在
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# 步骤1: 检查系统环境
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "步骤 1/6: 检查系统环境"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 检查是否在WSL中
if grep -qi microsoft /proc/version; then
    print_success "检测到WSL环境"
else
    print_warning "可能不在WSL环境中，但继续执行..."
fi

# 检查操作系统
if [ -f /etc/os-release ]; then
    . /etc/os-release
    print_info "操作系统: $NAME $VERSION"
else
    print_warning "无法检测操作系统版本"
fi

echo ""

# 步骤2: 检查并安装依赖
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "步骤 2/6: 检查并安装系统依赖"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 检查Python3
if command_exists python3; then
    PYTHON_VERSION=$(python3 --version)
    print_success "Python3已安装: $PYTHON_VERSION"
else
    print_info "安装Python3..."
    sudo apt update -qq
    sudo apt install -y python3 python3-pip
    print_success "Python3安装完成"
fi

# 检查pip3
if command_exists pip3; then
    PIP_VERSION=$(pip3 --version | awk '{print $2}')
    print_success "pip3已安装: $PIP_VERSION"
else
    print_info "安装pip3..."
    sudo apt install -y python3-pip
    print_success "pip3安装完成"
fi

# 检查Git
if command_exists git; then
    GIT_VERSION=$(git --version)
    print_success "Git已安装: $GIT_VERSION"
else
    print_info "安装Git..."
    sudo apt install -y git
    print_success "Git安装完成"
fi

echo ""

# 步骤3: 选择安装方法
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "步骤 3/6: 选择项目获取方法"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "请选择获取项目的方法:"
echo "  1) 从GitHub克隆（推荐）"
echo "  2) 从Windows文件系统复制"
echo "  3) 从压缩包解压"
echo ""
read -p "请输入选项 [1-3] (默认: 1): " INSTALL_METHOD
INSTALL_METHOD=${INSTALL_METHOD:-1}

# 创建项目目录
mkdir -p "$INSTALL_DIR"

case $INSTALL_METHOD in
    1)
        print_info "使用Git克隆方法..."
        echo ""
        
        # 检查项目是否已存在
        if [ -d "$PROJECT_DIR" ]; then
            print_warning "项目目录已存在: $PROJECT_DIR"
            read -p "是否删除并重新克隆? [y/N]: " CONFIRM
            if [[ $CONFIRM =~ ^[Yy]$ ]]; then
                rm -rf "$PROJECT_DIR"
                print_info "已删除旧目录"
            else
                print_error "安装已取消"
                exit 1
            fi
        fi
        
        # 克隆项目
        print_info "正在从GitHub克隆项目..."
        if git clone "$GITHUB_URL" "$PROJECT_DIR"; then
            print_success "项目克隆成功"
        else
            print_error "克隆失败，请检查网络连接"
            exit 1
        fi
        ;;
        
    2)
        print_info "使用文件系统复制方法..."
        echo ""
        read -p "请输入Windows中的项目路径 (例: /mnt/c/Users/YourName/Documents/BiYeSheji): " WINDOWS_PATH
        
        if [ ! -d "$WINDOWS_PATH" ]; then
            print_error "目录不存在: $WINDOWS_PATH"
            exit 1
        fi
        
        print_info "正在复制文件..."
        cp -r "$WINDOWS_PATH" "$INSTALL_DIR/"
        print_success "文件复制完成"
        ;;
        
    3)
        print_info "使用压缩包解压方法..."
        echo ""
        read -p "请输入ZIP文件路径 (例: /mnt/c/Users/YourName/Downloads/BiYeSheji.zip): " ZIP_PATH
        
        if [ ! -f "$ZIP_PATH" ]; then
            print_error "文件不存在: $ZIP_PATH"
            exit 1
        fi
        
        # 检查unzip是否安装
        if ! command_exists unzip; then
            print_info "安装unzip..."
            sudo apt install -y unzip
        fi
        
        print_info "正在解压文件..."
        unzip -q "$ZIP_PATH" -d "$INSTALL_DIR/"
        print_success "文件解压完成"
        ;;
        
    *)
        print_error "无效的选项"
        exit 1
        ;;
esac

echo ""

# 步骤4: 修复文件权限
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "步骤 4/6: 修复文件权限"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cd "$PROJECT_DIR"

print_info "修复Python文件权限..."
find . -type f -name "*.py" -exec chmod 644 {} \;
chmod +x persistent_server.py simple_demo.py 2>/dev/null || true

print_info "修复目录权限..."
find . -type d -exec chmod 755 {} \;

print_success "权限修复完成"
echo ""

# 步骤5: 安装Python依赖
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "步骤 5/6: 安装Python依赖"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -f "requirements.txt" ]; then
    print_info "正在安装Python依赖包..."
    echo ""
    
    # 升级pip
    pip3 install --upgrade pip -q
    
    # 安装依赖
    if pip3 install -r requirements.txt; then
        print_success "依赖安装完成"
    else
        print_warning "部分依赖安装失败，但继续..."
    fi
else
    print_warning "未找到requirements.txt"
fi

echo ""

# 步骤6: 验证安装
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "步骤 6/6: 验证安装"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 检查关键文件
print_info "检查项目文件..."

REQUIRED_FILES=(
    "README.md"
    "requirements.txt"
    "persistent_server.py"
    "simple_demo.py"
)

REQUIRED_DIRS=(
    "data_processing"
    "models"
    "visualization"
    "web_app"
    "tests"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        print_success "文件存在: $file"
    else
        print_error "文件缺失: $file"
    fi
done

for dir in "${REQUIRED_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        print_success "目录存在: $dir"
    else
        print_error "目录缺失: $dir"
    fi
done

# 检查Python包
echo ""
print_info "检查Python包..."

PACKAGES=("numpy" "gradio" "pillow")
for pkg in "${PACKAGES[@]}"; do
    if python3 -c "import $pkg" 2>/dev/null; then
        print_success "Python包已安装: $pkg"
    else
        print_warning "Python包未安装: $pkg"
    fi
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ 安装完成！"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
print_success "项目已成功安装到: $PROJECT_DIR"
echo ""
echo "📋 下一步操作:"
echo ""
echo "  1. 进入项目目录:"
echo "     cd $PROJECT_DIR"
echo ""
echo "  2. 运行演示服务器:"
echo "     python3 persistent_server.py"
echo ""
echo "  3. 在浏览器中访问:"
echo "     http://localhost:7860"
echo ""
echo "📚 更多信息:"
echo "  - 详细文档: cat WSL项目复制指南.md"
echo "  - 快速参考: cat WSL快速参考.txt"
echo "  - 项目说明: cat README.md"
echo ""
echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║                    祝您使用愉快！                                        ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
