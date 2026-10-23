#!/usr/bin/env bash
# verify_hf_ckpt.sh —— HF 权重目录静态校验（只读，不改任何文件）
#
# 用法: bash verify_hf_ckpt.sh <hf_ckpt_dir> [--strict-package] [--expect-sha256 FILE]
# 退出码: 0=PASS（无硬错误） / 1=FAIL（有硬错误） / 2=用法或路径错误
#
# 校验了什么（纯静态，不需要 torch / numpy / safetensors）
#   1) 结构：model.safetensors.index.json、config.json 存在，且 index 是合法 JSON
#   2) 分片清单：以 index.weight_map 声明的分片数为准（不再硬编码 6 片），
#      并与磁盘 *.safetensors 做集合比对 —— 缺片 / 多出的陈旧分片 / 改名都会报出来
#   3) 体积（与官方 _get_shard_info() 完全同口径）：逐片解析 safetensors 头，
#      用 data_offsets 求「数据区字节」，Σ数据区 必须与 index.metadata.total_size 精确相等。
#      官方 total_size = Σ numel×itemsize，是纯张量数据字节，不含「8 字节头长度 + JSON 头」；
#      所以绝不能拿文件大小去和 total_size 比（那会让完全正确的权重必然判 FAIL）。
#      文件字节数只做 ≥ total_size 的粗检；每片单独打印头部字节数，便于人看。
#   4) 分片内部一致性：数据区字节 == 头部张量声明的字节（可发现截断 / 尾部多余数据）
#   5) 张量名与 weight_map 一致：每片头部的张量名集合 == index 归到该片的张量名集合，
#      且 index 里没有「哪个分片都找不到」的张量（可发现 index 与分片内容错配）
#   6) 内容断言：config.json 合法 JSON 且精度口径为 float32（声明了才查，不符只告警）；
#      tokenizer 资产存在性（tokenizer_config.json + 词表之一）；
#      index.weight_map 非空、无重复键
#   7) sha256：逐文件打印；有基线（--expect-sha256 FILE，或同目录自动发现
#      SHA256SUMS / SHA256SUMS.txt / *.sha256）就逐项比对（兼容 CRLF、`sha256sum -b`
#      的 * 标记与 ./ 前缀）；sha256 计算出错、基线条目在本目录缺失都计入硬错误
#   8) 提交包整洁性：同级 DCP model/ 目录默认只告警并单独计数，
#      --strict-package 时才计入失败
#
# 没校验什么（只能靠 deploy server 严格加载兜底）
#   * 张量名 / 形状是否与模型结构对得上：语义错误的导出（名字错成另一个合法名字、
#     layout/维序错、该 covered 的 module 少了但 index 与分片自洽）在本脚本看来完全
#     「结构正确、字节正确」→ 本脚本一定 PASS，deploy 才会报错。
#     第 5 条只保证「index 与分片内容互相自洽」，不保证「与模型 __init__ 对得上」。
#   * 数值是否正确：导错了 step、EMA 与普通分支搞混、权重根本没初始化，本脚本都发现不了
#   * 同尺寸位翻转：没有基线时只打印 sha256 不比对，只能发现缺失 / 截断
#   * 张量 dtype 逐项核对（只读 config 的声明与分片头的 dtype 字段，不校验数值范围/是否 NaN）
#   * tokenizer 内容是否可用（只查文件存在，不校验 vocab 是否真能编码 decode）
#   * 分片顺序语义、shard_size 切分策略是否与官方一致
#   * 单文件（未分片、无 index）布局：本脚本要求存在 index，单文件布局会判 FAIL
#   * 磁盘空间 / 传输完整性之外的文件系统问题（稀疏文件等）
#
# 只读保证：仅以 'rb'/'r' 打开被校验目录内的文件，绝不写 / 改 / 删。
# 依赖：bash 4+、coreutils、python3 标准库。
set -uo pipefail

USAGE='用法: bash verify_hf_ckpt.sh <hf_ckpt_dir> [--strict-package] [--expect-sha256 FILE]'

[ $# -ge 1 ] || { printf '%s\n' "$USAGE" >&2; exit 2; }
: "${1:?$USAGE}"
D=''
strict_flag=0
expect=''
expect_given=0
while [ $# -gt 0 ]; do
  case "$1" in
    --strict-package) strict_flag=1; shift ;;
    --expect-sha256)
      [ $# -ge 2 ] || { printf '选项 --expect-sha256 缺少文件参数\n%s\n' "$USAGE" >&2; exit 2; }
      expect=$2; expect_given=1; shift 2 ;;
    --expect-sha256=*) expect=${1#*=}; expect_given=1; shift ;;
    -h|--help) printf '%s\n' "$USAGE"; exit 0 ;;
    -*) printf '未知选项: %s\n%s\n' "$1" "$USAGE" >&2; exit 2 ;;
    *)
      if [ -z "$D" ]; then D=$1; else printf '多余参数: %s\n%s\n' "$1" "$USAGE" >&2; exit 2; fi
      shift ;;
  esac
done
[ -n "$D" ] || { printf '%s\n' "$USAGE" >&2; exit 2; }
if [ "$expect_given" -eq 1 ] && [ -z "$expect" ]; then
  printf '选项 --expect-sha256 的文件参数为空\n%s\n' "$USAGE" >&2
  exit 2
fi
D=${D%/}; [ -n "$D" ] || D=/
[ -d "$D" ] || { echo "目录不存在: $D"; exit 2; }
if [ -n "$expect" ] && [ ! -f "$expect" ]; then
  echo "基线文件不存在: $expect"
  exit 2
fi
command -v python3 >/dev/null 2>&1 || { echo "需要 python3（未找到）"; exit 2; }

rc=0
python3 - "$D" "$strict_flag" "$expect" <<'PY' || rc=$?
import glob
import hashlib
import json
import os
import sys
import unicodedata

D = sys.argv[1]
STRICT = sys.argv[2] == '1'
EXPECT = sys.argv[3]

COL = 38
errs = []
warns = []


def dw(s):
    """显示宽度：CJK 宽字符算 2 列，其余算 1 列（近似 wcwidth）。"""
    w = 0
    for ch in s:
        if unicodedata.combining(ch):
            continue
        w += 2 if unicodedata.east_asian_width(ch) in ('W', 'F') else 1
    return w


def say(label, value):
    print(label + ' ' * max(1, COL - dw(label)) + str(value))


def sec(title):
    print('--- %s ---' % title)


def err(msg):
    errs.append(msg)
    say('错误', msg)


def warn(msg):
    warns.append(msg)
    say('警告', msg)


def table(rows, headers):
    allrows = [headers] + rows
    widths = [max(dw(r[i]) for r in allrows) for i in range(len(headers))]
    for r in allrows:
        print('  '.join(c + ' ' * (widths[i] - dw(c)) for i, c in enumerate(r)).rstrip())


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def parse_shard(p):
    """解析 safetensors 头。
    返回 (文件字节, 头部字节含8字节长度, Σ张量字节, 张量数, 最大结束偏移, 张量名集合)。
    """
    size = os.path.getsize(p)
    with open(p, 'rb') as f:
        head = f.read(8)
        if len(head) != 8:
            raise ValueError('文件不足 8 字节，读不到头长度')
        n = int.from_bytes(head, 'little')
        if n <= 0:
            raise ValueError('头部长度非法: %d' % n)
        if n > size - 8:
            raise ValueError('头部长度 %d 超过文件剩余 %d 字节' % (n, size - 8))
        raw = f.read(n)
    if len(raw) != n:
        raise ValueError('头部被截断')
    try:
        hdr = json.loads(raw.decode('utf-8'))
    except Exception as e:
        raise ValueError('头部不是合法 JSON: %s' % e)
    if not isinstance(hdr, dict):
        raise ValueError('头部顶层不是 JSON 对象')
    used, end_max, nten = 0, 0, 0
    tnames = set()
    for k, v in hdr.items():
        if k == '__metadata__':
            continue
        tnames.add(k)
        if not isinstance(v, dict) or 'data_offsets' not in v:
            raise ValueError('张量 %s 缺少 data_offsets' % k)
        o = v['data_offsets']
        if not (isinstance(o, list) and len(o) == 2):
            raise ValueError('张量 %s 的 data_offsets 非法' % k)
        b, e = int(o[0]), int(o[1])
        if e < b:
            raise ValueError('张量 %s 的 data_offsets 反向' % k)
        used += e - b
        end_max = max(end_max, e)
        nten += 1
    if nten == 0:
        raise ValueError('头部没有任何张量')
    return size, 8 + n, used, nten, end_max, tnames


def main():
    print('校验目录 %s（只读）' % D)
    print('模式 %s' % ('严格提交包（--strict-package）' if STRICT else '普通'))

    idx_name = 'model.safetensors.index.json'
    cfg_name = 'config.json'

    # ---------------- 1. 结构 ----------------
    sec('结构')
    for f in (idx_name, cfg_name):
        p = os.path.join(D, f)
        if os.path.isfile(p):
            say(f, '存在')
        else:
            say(f, '缺失')
            err('缺失必需文件 %s' % f)

    idx = None
    dup_keys = []
    wm = {}
    declared = []
    total_meta = None
    p_idx = os.path.join(D, idx_name)
    if os.path.isfile(p_idx):

        def hook(pairs):
            seen = set()
            for k, _ in pairs:
                if k in seen:
                    dup_keys.append(k)
                seen.add(k)
            return dict(pairs)

        try:
            with open(p_idx, 'r', encoding='utf-8') as f:
                idx = json.load(f, object_pairs_hook=hook)
        except Exception as e:
            err('%s 不是合法 JSON：%s' % (idx_name, e))
            idx = None

    if idx is not None:
        if dup_keys:
            uniq = sorted(set(dup_keys))
            warn('index 中存在重复键 %d 个（JSON 后者覆盖前者）：%s'
                 % (len(uniq), ', '.join(uniq[:5])))
        if not isinstance(idx, dict):
            err('index 顶层不是 JSON 对象（无法解析 weight_map）')
        else:
            _wm = idx.get('weight_map')
            if not isinstance(_wm, dict) or not _wm:
                err('index.weight_map 缺失或为空（无法确定分片清单）')
            else:
                wm = _wm
                declared = sorted({str(v) for v in wm.values()})
            md = idx.get('metadata')
            if isinstance(md, dict):
                total_meta = md.get('total_size')

    disk = sorted(f for f in os.listdir(D)
                  if f.endswith('.safetensors') and os.path.isfile(os.path.join(D, f)))
    say('磁盘 *.safetensors 文件数', len(disk))
    say('index 声明的分片数', len(declared) if wm else '无法确定（weight_map 不可用）')
    say('index 中张量条目数', len(wm) if wm else '无法确定（weight_map 不可用）')
    if not disk:
        err('目录内没有任何 *.safetensors 分片')

    if wm:
        missing = [f for f in declared if f not in disk]
        extra = [f for f in disk if f not in declared]
        say('分片集合比对（磁盘 vs index）', '一致' if not (missing or extra) else '不一致')
        if missing:
            err('缺失分片 %d 个：%s' % (len(missing), ', '.join(missing)))
        if extra:
            err('多出 index 未声明的分片 %d 个：%s' % (len(extra), ', '.join(extra)))
    else:
        say('分片集合比对（磁盘 vs index）', '跳过（index 不可用）')

    # ---------------- 2. 体积（数据区口径） ----------------
    sec('体积（数据区口径，与官方 _get_shard_info 一致）')
    if not declared:
        say('体积校验', '跳过（无 index.weight_map 可用的分片清单，无法确定应有体积）')
    rows = []
    shard_errs = []
    shard_tensors = {}
    tot_file = tot_head = tot_used = tot_region = 0
    for i, name in enumerate(declared, 1):
        p = os.path.join(D, name)
        if not os.path.isfile(p):
            continue
        try:
            size, head, used, nten, end_max, tnames = parse_shard(p)
        except Exception as e:
            shard_errs.append('%s 头部解析失败：%s' % (name, e))
            rows.append([str(i), name, '-', '-', '-', '-', '解析失败'])
            continue
        shard_tensors[name] = (tnames, used)
        region = size - head
        tot_file += size
        tot_head += head
        tot_used += used
        tot_region += region
        status = 'OK'
        if region < used or end_max > region:
            status = '数据区不足(截断?)'
            shard_errs.append('%s 数据区 %d B < 头部张量声明所需 %d B（文件被截断，或张量数据没写完）'
                              % (name, region, used))
        elif region > used:
            status = '尾部多余字节'
            shard_errs.append('%s 数据区 %d B > 头部张量声明所需 %d B（尾部多出 %d B）'
                              % (name, region, used, region - used))
        rows.append([str(i), name, str(head), str(region), str(used), str(size), status])
    if rows:
        table(rows, ['#', '分片文件', '头部字节', '数据区字节', '张量所需字节', '文件字节', '状态'])
    for m in shard_errs:
        err(m)
    if declared:
        say('分片数据区字节合计', tot_region)
        say('张量所需字节合计（Σ data_offsets）', tot_used)
        say('分片内部自洽（数据区 vs 张量声明）', '一致' if tot_region == tot_used else '不一致')
        say('分片头部字节合计（8B 长度 + JSON 头）', tot_head)
        say('磁盘文件字节合计', tot_file)

    ts = None
    if total_meta is None:
        if idx is not None and isinstance(idx, dict):
            warn('index.metadata.total_size 未声明，跳过体积精确比对（只能靠分片内部一致性）')
    else:
        try:
            ts = int(total_meta)
        except Exception:
            err('index.metadata.total_size 不是整数：%r' % (total_meta,))

    if ts is not None:
        eq = (ts == tot_region)
        say('体积比对 Σ数据区 vs total_size',
            '%d vs %d → %s' % (tot_region, ts, 'OK' if eq else '不一致'))
        if not eq:
            err('体积不一致：实测数据区 %d B，index total_size %d B，差 %+d B'
                % (tot_region, ts, tot_region - ts))
        ge = tot_file >= ts
        say('粗检 文件字节 ≥ total_size',
            '%d ≥ %d → %s' % (tot_file, ts, 'OK' if ge else '不合格'))
        if not ge:
            err('磁盘文件字节合计 %d B 小于 total_size %d B' % (tot_file, ts))

    # ---------------- 2b. 张量名与 weight_map 的一致性 ----------------
    if wm and not missing and not extra and shard_tensors:
        by_shard = {}
        for tn, fn in wm.items():
            by_shard.setdefault(fn, set()).add(tn)
        problems = []
        seen_all = set()
        for name, (tnames, _used) in sorted(shard_tensors.items()):
            expect = by_shard.get(name, set())
            seen_all |= tnames
            lack = expect - tnames
            surplus = tnames - expect
            if lack:
                problems.append('%s 缺少 index 归到本片的张量 %d 个：%s'
                                % (name, len(lack), ', '.join(sorted(lack)[:3])))
            if surplus:
                problems.append('%s 含 index 未归到本片的张量 %d 个：%s'
                                % (name, len(surplus), ', '.join(sorted(surplus)[:3])))
        lost = set(wm) - seen_all
        if lost:
            problems.append('index 声明但所有分片头部都找不到的张量 %d 个：%s'
                            % (len(lost), ', '.join(sorted(lost)[:3])))
        say('张量名与 weight_map 一致', '是' if not problems else '否')
        for m in problems[:10]:
            err(m)
        if len(problems) > 10:
            err('…… 同类问题还有 %d 条，已省略' % (len(problems) - 10))
    else:
        say('张量名与 weight_map 一致', '跳过（分片清单不完整，先修上面的缺/多片问题）')

    # ---------------- 3. 内容断言 ----------------
    sec('内容断言')
    cfg, cfg_err = None, None
    p_cfg = os.path.join(D, cfg_name)
    if os.path.isfile(p_cfg):
        try:
            with open(p_cfg, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
        except Exception as e:
            cfg_err = e
    if cfg_err is not None:
        say('config.json 合法 JSON', '否')
        err('config.json 不是合法 JSON：%s' % cfg_err)
    elif os.path.isfile(p_cfg):
        say('config.json 合法 JSON', '是（顶层不是对象）' if not isinstance(cfg, dict) else '是')
    if os.path.isfile(p_cfg) and cfg_err is None and not isinstance(cfg, dict):
        err('config.json 顶层不是 JSON 对象')
    cfg_ok = isinstance(cfg, dict)
    dt = cfg.get('torch_dtype', cfg.get('dtype')) if cfg_ok else None
    if dt is not None:
        say('config 声明精度', dt)
        norm = str(dt).lower().replace('torch.', '').strip()
        if norm not in ('float32', 'fp32', 'f32'):
            warn('config 声明精度 %s，与官方提交口径「F32 / ≈25.5 GB」不符'
                 '（若确为 bf16 导出，体积减半属正常，请确认这是你要的那份）' % dt)
    else:
        say('config 声明精度', '未声明' if cfg_ok else '无法判定（config 不可解析）')

    tok_cfg = os.path.join(D, 'tokenizer_config.json')
    say('tokenizer_config.json', '存在' if os.path.isfile(tok_cfg) else '缺失')
    if not os.path.isfile(tok_cfg):
        err('缺失 tokenizer_config.json（多半是导出时 --assets-from 漏拷，到 deploy 才炸）')
    vocab_cands = ('tokenizer.json', 'vocab.json', 'tokenizer.model', 'spiece.model')
    found_vec = [n for n in vocab_cands if os.path.isfile(os.path.join(D, n))]
    say('tokenizer 词表资产', ', '.join(found_vec) if found_vec else '缺失')
    if not found_vec:
        err('缺失词表文件（%s 至少要有一个；--assets-from 漏拷会静默通过导出）'
            % ' / '.join(vocab_cands))

    # ---------------- 4. sha256 ----------------
    sec('sha256')
    cand = list(disk) + [idx_name, cfg_name, 'tokenizer_config.json', 'tokenizer.json',
                         'vocab.json', 'tokenizer.model', 'generation_config.json']
    names = []
    for n in cand:
        if n not in names and os.path.isfile(os.path.join(D, n)):
            names.append(n)
    names.sort()
    hashes = {}
    for n in names:
        try:
            h = sha256_file(os.path.join(D, n))
        except Exception as e:
            err('%s sha256 计算失败：%s' % (n, e))
            continue
        hashes[n] = h
        print('%s  %s' % (h, n))

    # 基线：显式指定 > 同目录自动发现
    base, base_src = EXPECT, '--expect-sha256'
    if not base:
        for c in ('SHA256SUMS', 'SHA256SUMS.txt', 'sha256sums.txt', 'SHA256SUMS.sha256'):
            if os.path.isfile(os.path.join(D, c)):
                base, base_src = os.path.join(D, c), '同目录自动发现'
                break
        if not base:
            g = sorted(glob.glob(os.path.join(D, '*.sha256')) +
                       glob.glob(os.path.join(D, '*.sha256sum')))
            if len(g) > 1:
                warn('同目录发现多个 *.sha256 基线，只用第一个：%s' % g[0])
            if g:
                base, base_src = g[0], '同目录自动发现'

    if base:
        say('基线文件', '%s（%s）' % (base, base_src))
        entries, bad_lines = [], []
        try:
            with open(base, 'r', encoding='utf-8', errors='replace') as f:
                for ln in f:
                    s = ln.strip()
                    if not s or s.startswith('#'):
                        continue
                    parts = s.split(None, 1)
                    if len(parts) != 2:
                        bad_lines.append(s)
                        continue
                    h, fn = parts[0].lower(), parts[1].strip()
                    if len(h) != 64 or any(ch not in '0123456789abcdef' for ch in h):
                        bad_lines.append(s)
                        continue
                    entries.append((fn.lstrip('*').strip().replace('\\', '/'), h))
        except Exception as e:
            err('基线文件无法读取：%s' % e)
        if bad_lines:
            err('基线中有 %d 行格式无法解析（示例：%s）' % (len(bad_lines), bad_lines[0][:60]))
        ok_n = bad_n = 0
        droot = os.path.realpath(D)
        for fn, h in entries:
            p = os.path.join(D, fn)
            if not os.path.isfile(p):
                p = os.path.join(D, os.path.basename(fn))
            real = os.path.realpath(p)
            if not (real == droot or real.startswith(droot + os.sep)):
                err('基线条目指向校验目录之外，已忽略：%s' % fn)
                bad_n += 1
                continue
            rel = os.path.relpath(p, D)
            if not os.path.isfile(p):
                err('基线列出了目录中不存在的文件：%s' % fn)
                bad_n += 1
                continue
            try:
                got = hashes.get(rel) or sha256_file(p)
            except Exception as e:
                err('%s sha256 计算失败：%s' % (rel, e))
                bad_n += 1
                continue
            hashes[rel] = got
            if got == h:
                ok_n += 1
            else:
                bad_n += 1
                err('sha256 不一致：%s（基线 %s… 实测 %s…）' % (fn, h[:12], got[:12]))
        say('基线比对', '%d 项一致 / %d 项不一致（基线共 %d 项）'
            % (ok_n, bad_n, len(entries)))
        if not entries:
            err('基线文件为空或全部行无法解析')
    else:
        say('基线比对', '未提供')
        print('!!! 基线比对: 未提供（--expect-sha256）—— '
              '只能发现缺失/截断，无法发现同尺寸位翻转')

    # ---------------- 5. 提交包整洁性 ----------------
    sec('提交包整洁性')
    parent = os.path.dirname(os.path.realpath(D))
    dcp = os.path.join(parent, 'model')
    if os.path.isdir(dcp):
        say('同级 DCP model/ 目录', '存在：%s' % dcp)
        if STRICT:
            err('同级存在 DCP model/ 目录，--strict-package 下视为打包不合格（应移出 03_模型权重）')
        else:
            warn('同级存在 DCP model/ 目录，提交前应移出 03_模型权重'
                 '（与权重完整性无关，默认不计入失败）')
    else:
        say('同级 DCP model/ 目录', '无')

    # ---------------- 汇总 ----------------
    print('---')
    say('硬错误数', len(errs))
    say('警告数', len(warns))
    if warns:
        say('提示', '有警告但无硬错误时仍判 PASS；加 --strict-package 可把打包告警升级为失败')
    return 0 if not errs else 1


try:
    sys.exit(main())
except SystemExit:
    raise
except BrokenPipeError:
    sys.exit(1)
except Exception as e:
    print('错误  校验过程异常：%s: %s' % (type(e).__name__, e))
    if os.environ.get('VFC_DEBUG'):
        import traceback
        traceback.print_exc()
    sys.exit(1)
PY

echo '---'
if [ "$rc" -eq 0 ]; then
  echo "RESULT: PASS（结构+体积+分片齐全）"
else
  echo "RESULT: FAIL（见上）"
fi
echo "注意：以上为静态校验；仍需官方 deploy server 严格加载一次才算通过。"
exit "$rc"
