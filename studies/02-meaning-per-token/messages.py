"""Synthetic task.create messages rendered in 5 formats. All 5 fields uniform & independent."""
import itertools, json, os, random, sys
# abios_refs needs the abios encoder: set ABIOS_DIR to an abios checkout at commit 6da3a64.
if os.environ.get("ABIOS_DIR"):
    sys.path.insert(0, os.path.join(os.environ["ABIOS_DIR"], "scripts", "protean-abios"))
TITLES = [("Fix login timeout","修复登录超时","login-timeout"),("Add CSV export","添加CSV导出","csv-export"),
 ("Refactor payment retries","重构支付重试","pay-retry"),("Speed up search index","加快搜索索引","search-speed"),
 ("Remove dead feature flags","删除无用的功能开关","dead-flags"),("Upgrade database driver","升级数据库驱动","db-driver"),
 ("Translate settings page","翻译设置页面","settings-i18n"),("Rotate API keys","轮换API密钥","key-rotate"),
 ("Fix memory leak in worker","修复工作进程内存泄漏","worker-leak"),("Add rate limiting","添加限流","rate-limit"),
 ("Improve error messages","改进错误信息","error-msgs"),("Write onboarding guide","编写入门指南","onboard-doc")]
PRI = [("low","低","↓"),("normal","普通","·"),("high","高","↑"),("urgent","紧急","‼")]
SCOPE = [("auth","认证","au"),("billing","计费","bi"),("search","搜索","se"),("api","接口","ap"),("ui","界面","ui"),("db","数据库","db")]
CRIT = [("tests pass","tests pass","测试通过","T"),("no regressions","there are no regressions","无回归","R"),
 ("human review","a human reviews it","人工审核通过","H"),("bench within 5%","benchmarks stay within 5%","基准测试波动不超过5%","B"),
 ("docs updated","the docs are updated","文档已更新","D")]
DEP = [False, True]
FIELDS = {"title": len(TITLES), "priority": len(PRI), "scope": len(SCOPE), "criteria": len(CRIT), "dependency": 2}
PID, TID = "a1" * 16, "b2" * 16

def make(n=600, seed=0):
    rng = random.Random(seed)
    allc = list(itertools.product(range(12), range(4), range(6), range(5), range(2)))
    rng.shuffle(allc)
    return [dict(zip(FIELDS, c)) | {"tmpl": rng.randrange(3)} for c in allc[:n]]

def english(m):
    t, p, s, c, d = TITLES[m["title"]][0], PRI[m["priority"]][0], SCOPE[m["scope"]][0], CRIT[m["criteria"]][1], m["dependency"]
    if m["tmpl"] == 0:
        return f'Create a new task in project p titled "{t}". Priority: {p}. Scope: the {s} module. It is done when {c}.' + (" It depends on task t." if d else "")
    if m["tmpl"] == 1:
        return f'Please open a {p}-priority task, "{t}", touching {s}. Acceptance: {c}.' + (" Blocked by task t." if d else " No dependencies.")
    return f"New task for {s}: {t}. {p.capitalize()} priority; finish when {c}." + (" Waits on task t." if d else "")

def json_compact(m):
    o = {"action": "task.create", "project_id": "p", "title": TITLES[m["title"]][0], "priority": PRI[m["priority"]][0],
         "scope": [SCOPE[m["scope"]][0]], "criteria": [CRIT[m["criteria"]][0]], "dependencies": ["t"] if m["dependency"] else []}
    return json.dumps(o, separators=(",", ":"), ensure_ascii=False)

_enc = None
def abios_refs(m):
    """Real ABIOS/3 reference encoder, refs family (tag E), fresh session per message."""
    global _enc
    from encoders import all_encoders, fresh_session
    if _enc is None: _enc = all_encoders()["refs" if "refs" in all_encoders() else "E"]
    pl = dict(project_id=PID, title=TITLES[m["title"]][0], criteria=[CRIT[m["criteria"]][0]], scope=[SCOPE[m["scope"]][0]],
              priority=PRI[m["priority"]][0], dependencies=[TID] if m["dependency"] else [])
    return _enc.encode("task.create", pl, fresh_session())

def chinese(m):
    t, p, s, c, d = TITLES[m["title"]][1], PRI[m["priority"]][1], SCOPE[m["scope"]][1], CRIT[m["criteria"]][2], m["dependency"]
    if m["tmpl"] == 1:
        return f"请新建一个{p}优先级任务“{t}”，涉及{s}模块。验收：{c}。" + ("依赖任务t。" if d else "无依赖。")
    return f"在项目p中新建任务“{t}”。优先级：{p}。范围：{s}模块。完成标准：{c}。" + ("依赖任务t。" if d else "")

def glyph(m):
    """My symbolic dialect 'G1': +⟨slug⟩<pri glyph>@<scope2>✓<crit letter>[←t]"""
    return (f"+⟨{TITLES[m['title']][2]}⟩{PRI[m['priority']][2]}@{SCOPE[m['scope']][2]}✓{CRIT[m['criteria']][3]}"
            + ("←t" if m["dependency"] else ""))

GLYPH_LEGEND = ("Legend for G1 lines: '+' = create task; ⟨slug⟩ = short title; ↓ low, · normal, ↑ high, ‼ urgent priority; "
  "@au auth, @bi billing, @se search, @ap api, @ui ui, @db db scope; ✓T tests pass, ✓R no regressions, ✓H human review, "
  "✓B bench within 5%, ✓D docs updated; ←t = depends on task t (absent = no dependency).")
ABIOS_LEGEND = ("ABIOS/3 line: tag+command (Ecta = refs-format task.create), then values in fixed order: criteria, dependencies, "
  "description, priority, project_id, scope, title. p = project ref; [] = no dependencies.")
FORMATS = {"english": english, "json": json_compact, "abios_refs": abios_refs, "chinese": chinese, "glyph_g1": glyph}

if __name__ == "__main__":
    for m in make(4, 1):
        for k, f in FORMATS.items(): print(k, "|", f(m))
