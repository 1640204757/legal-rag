// 页面状态只保存在当前标签页；会话编号用于隔离服务端内存任务。
let token = "",
  sessionId = "",
  requestId = "",
  busy = false,
  epoch = 0,
  lastResult = null,
  showAll = false;

const $ = (id) => document.getElementById(id);

// 全局浮动轻提示（Toast）
let toastTimer = null;
function showToast(msg) {
  const t = $("toast");
  if (!t) return;
  t.textContent = msg;
  t.hidden = false;
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    t.hidden = true;
  }, 2200);
}

// 统一调用本机接口，附带本页令牌；错误以可读提示展示。
async function api(path, body) {
  const r = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Local-Token": token },
    body: JSON.stringify(body || {}),
  });
  const data = await r.json();
  if (!r.ok) throw new Error(data.error || "请求失败，请刷新页面重试");
  return data;
}

function progress(text, error = false) {
  const p = $("progress");
  if (!p) return;
  p.textContent = text;
  p.className = "progress-bar" + (error ? " error" : "");
}

function setBusy(on) {
  busy = on;
  $("send").disabled = on;
  $("stop").hidden = !on;
  $("question").disabled = on;
}

// 自动平滑滚动到底部
function scrollToBottom() {
  const vp = $("chat-viewport");
  if (vp) {
    setTimeout(() => {
      vp.scrollTop = vp.scrollHeight;
    }, 40);
  }
}

// 安全链接校验
function safeLink(url) {
  try {
    const u = new URL(url);
    return ["https:", "http:"].includes(u.protocol) ? u.href : null;
  } catch {
    return null;
  }
}

// 输入目录URL，返回安全官方HTTPS链接或null；不发请求，目录提示不替第三方站点背书。
function safeOfficialLink(url) {
  const link = safeLink(url);
  if (!link || typeof url !== "string" || /[\x00-\x20\x7f]/.test(url)) return null;
  try {
    const u = new URL(link);
    const authority = url.split("/")[2] || "";
    return u.protocol === "https:" &&
      u.hostname.endsWith(".gov.cn") &&
      !authority.includes("@") &&
      !u.username &&
      !u.password &&
      (!u.port || u.port === "443")
      ? link
      : null;
  } catch {
    return null;
  }
}

// 高亮并定位到右侧来源卡片
function highlightSource(materialIdOrTitle) {
  const aside = $("reference-aside");
  if (aside && window.innerWidth <= 820) {
    aside.classList.add("active");
  }
  const cards = document.querySelectorAll(".source");
  for (const card of cards) {
    const text = card.textContent || "";
    if (text.includes(materialIdOrTitle)) {
      card.scrollIntoView({ behavior: "smooth", block: "center" });
      card.classList.add("highlight");
      setTimeout(() => card.classList.remove("highlight"), 2500);
      showToast(`已在右侧定位依据：${materialIdOrTitle}`);
      return;
    }
  }
  showToast(`未在当前标签页中找到 ${materialIdOrTitle}，可切换至“检索候选”`);
}

// 交互式渲染助手回答（解析追问、处理方向、证据清单与不确定性）
function renderAssistantContent(body, text) {
  const sections = text.split(/(?=^##\s*.+$)/m);
  
  for (const sec of sections) {
    const trimmed = sec.trim();
    if (!trimmed) continue;
    
    const match = trimmed.match(/^##\s*([^\n]+)\n?([\s\S]*)$/);
    if (!match) {
      const p = document.createElement("p");
      p.className = "plain-text";
      p.textContent = trimmed;
      body.append(p);
      continue;
    }
    
    const title = match[1].trim();
    const content = match[2].trim();
    
    if (title.includes("追问")) {
      // 1. 追问卡片：提供一键填入追问回复
      const card = document.createElement("div");
      card.className = "section-card section-inquiry";
      
      const head = document.createElement("div");
      head.className = "section-title";
      head.textContent = "🔍 需补充核实的事实细节";
      card.append(head);
      
      const list = document.createElement("div");
      list.className = "inquiry-list";
      
      const lines = content.split("\n").map(l => l.trim()).filter(Boolean);
      for (const line of lines) {
        const item = document.createElement("div");
        item.className = "inquiry-item";
        
        const txt = document.createElement("span");
        txt.className = "inquiry-text";
        txt.textContent = line;
        
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "inquiry-btn";
        btn.textContent = "💬 补充细节";
        btn.onclick = () => {
          const q = $("question");
          const prompt = `针对问题【${line.replace(/^[\d一二三四五六七八九十、.\s]+/, "")}】，我的补充是：`;
          q.value = prompt;
          q.focus();
          q.setSelectionRange(prompt.length, prompt.length);
          showToast("已将追问内容填入输入框，请补充事实");
        };
        
        item.append(txt, btn);
        list.append(item);
      }
      card.append(list);
      body.append(card);
      
    } else if (title.includes("处理方向") || title.includes("处理步骤")) {
      // 2. 处理方向：流程时间线卡片，法条高亮药丸
      const card = document.createElement("div");
      card.className = "section-card section-roadmap";
      
      const head = document.createElement("div");
      head.className = "section-title";
      head.textContent = "🧭 推荐维权处理路径与法律步骤";
      card.append(head);
      
      const stepper = document.createElement("div");
      stepper.className = "stepper-flow";
      
      const rawSteps = content.split(/(?=^(?:第[一二三四五六七八九十]+步|步骤[一二三四五六七八九十\d]+|\d+[.、]))/m)
        .map(s => s.trim()).filter(Boolean);
      
      const steps = rawSteps.length > 0 ? rawSteps : [content];
      let stepIndex = 1;
      
      for (const s of steps) {
        const stepCard = document.createElement("div");
        stepCard.className = "step-card";
        
        const badge = document.createElement("div");
        badge.className = "step-badge";
        badge.textContent = String(stepIndex++).padStart(2, "0");
        
        const desc = document.createElement("div");
        desc.className = "step-content";
        
        const parts = s.split(/(《[^》]+》第[零〇一二三四五六七八九十百千万两\d]+条(?:之[零〇一二三四五六七八九十百千万两\d]+)?|《[^》]+》|M\d+)/g);
        for (const p of parts) {
          if (!p) continue;
          if (p.startsWith("《") || /^M\d+$/.test(p)) {
            const pill = document.createElement("button");
            pill.type = "button";
            pill.className = "law-pill";
            pill.textContent = p;
            pill.title = "点击在右侧定位查看法条原文";
            pill.onclick = () => highlightSource(p);
            desc.append(pill);
          } else {
            desc.append(document.createTextNode(p));
          }
        }
        
        stepCard.append(badge, desc);
        stepper.append(stepCard);
      }
      card.append(stepper);
      body.append(card);
      
    } else if (title.includes("证据清单") || title.includes("证据")) {
      // 3. 证据清单：可直接勾选标记的 Checklist + 进度计数
      const card = document.createElement("div");
      card.className = "section-card section-checklist";
      
      const head = document.createElement("div");
      head.className = "section-title";
      
      const titleSpan = document.createElement("span");
      titleSpan.textContent = "📋 关键证据准备清单 (点击可勾选已收集项)";
      
      const counterSpan = document.createElement("span");
      counterSpan.style.fontSize = "12px";
      counterSpan.style.color = "#059669";
      counterSpan.style.fontWeight = "600";
      counterSpan.style.marginLeft = "auto";
      
      head.append(titleSpan, counterSpan);
      card.append(head);
      
      const group = document.createElement("div");
      group.className = "checklist-group";
      
      const lines = content.split("\n").map(l => l.trim()).filter(Boolean);
      let checkedCount = 0;
      const totalCount = lines.length;
      
      const updateCounter = () => {
        counterSpan.textContent = totalCount ? `(已收集 ${checkedCount}/${totalCount})` : "";
      };
      updateCounter();
      
      for (const line of lines) {
        const item = document.createElement("label");
        item.className = "checklist-item";
        
        const cb = document.createElement("input");
        cb.type = "checkbox";
        cb.className = "checklist-checkbox";
        cb.onchange = () => {
          if (cb.checked) {
            checkedCount++;
            item.classList.add("checked");
          } else {
            checkedCount = Math.max(0, checkedCount - 1);
            item.classList.remove("checked");
          }
          updateCounter();
        };
        
        const txt = document.createElement("span");
        txt.className = "checklist-text";
        txt.textContent = line.replace(/^[-*•\d.、\s]+/, "");
        
        item.append(cb, txt);
        group.append(item);
      }
      card.append(group);
      body.append(card);
      
    } else if (title.includes("不确定性") || title.includes("风险")) {
      // 4. 不确定性与风险提示
      const card = document.createElement("div");
      card.className = "section-card section-uncertainty";
      
      const head = document.createElement("div");
      head.className = "section-title";
      head.textContent = "⚠️ 法律适用风险与事实不确定性";
      card.append(head);
      
      const b = document.createElement("div");
      b.className = "uncertainty-body";
      b.textContent = content;
      card.append(b);
      body.append(card);
      
    } else {
      // 其他带标题段落
      const card = document.createElement("div");
      card.className = "section-card";
      
      const head = document.createElement("div");
      head.className = "section-title";
      head.textContent = title;
      card.append(head);
      
      const b = document.createElement("div");
      b.className = "plain-text";
      b.textContent = content;
      card.append(b);
      body.append(card);
    }
  }
}

// 安全显示问答正文：用文本节点与结构化卡片，不执行模型生成的HTML。
function bubble(role, text) {
  const box = document.createElement("div");
  box.className = "bubble " + role;
  
  if (role === "user") {
    const who = document.createElement("div");
    who.className = "speaker";
    who.textContent = "👤 你描述的案情情况";
    box.append(who);
    
    const body = document.createElement("div");
    body.className = "body";
    body.textContent = text;
    box.append(body);
  } else {
    // 助手气泡头部操作栏：身份标识 + 复制/导出按钮
    const toolbar = document.createElement("div");
    toolbar.className = "assistant-toolbar";
    
    const identity = document.createElement("div");
    identity.className = "assistant-identity";
    
    const icon = document.createElement("span");
    icon.className = "bot-icon";
    icon.textContent = "法";
    
    const name = document.createElement("span");
    name.className = "assistant-name";
    name.textContent = "知法智能助手 · 初步分析";
    
    identity.append(icon, name);
    
    const tools = document.createElement("div");
    tools.className = "assistant-tools";
    
    const copyBtn = document.createElement("button");
    copyBtn.type = "button";
    copyBtn.className = "tool-btn";
    copyBtn.textContent = "📋 复制方案";
    copyBtn.onclick = () => {
      navigator.clipboard.writeText(text).then(() => {
        showToast("已成功复制分析方案与证据清单到剪贴板！");
      }).catch(() => {
        showToast("复制失败，请手动选取文本复制");
      });
    };
    
    const printBtn = document.createElement("button");
    printBtn.type = "button";
    printBtn.className = "tool-btn";
    printBtn.textContent = "🖨️ 打印备忘";
    printBtn.onclick = () => window.print();
    
    tools.append(copyBtn, printBtn);
    toolbar.append(identity, tools);
    box.append(toolbar);
    
    const body = document.createElement("div");
    body.className = "body";
    renderAssistantContent(body, text);
    box.append(body);
  }
  
  $("messages").append(box);
  scrollToBottom();
  return box;
}

// 每张依据都标明采集版本。版本号来自已保存快照，不表示已经在线核对仍然有效。
function currencyNote(versionId) {
  const version = versionId || "版本待核实";
  return "现行性：采集版本 " + version + "，未逐条在线复核是否仍然有效；未核个案适用。";
}

// 把尾部候选在两路里的名次写成"语义第N名 · 关键词第N名"。
function routeText(routes) {
  const parts = [];
  if (routes && routes.dense) parts.push(`语义第${routes.dense}名`);
  if (routes && routes.bm25) parts.push(`关键词第${routes.bm25}名`);
  return parts.join(" · ") || "来源名次未知";
}

// "更多候选"默认收起：只列法名、条号和来自哪一路第几名，点开看原文；这些条文没有送入模型。
function tailSection(rows) {
  const box = document.createElement("details");
  box.className = "tail-candidates";
  const summary = document.createElement("summary");
  summary.textContent = `更多候选（${rows.length}条，未送入模型，仅供参考）`;
  box.append(summary);
  for (const r of rows) {
    const item = document.createElement("article");
    item.className = "source tail";
    const title = document.createElement("h3");
    title.textContent = r.title;
    const loc = document.createElement("p");
    loc.textContent = (r.locator || "") + " · " + routeText(r.routes);
    const currency = document.createElement("p");
    currency.className = "currency";
    currency.textContent = currencyNote(r.version_id);
    const details = document.createElement("details");
    const more = document.createElement("summary");
    more.textContent = "展开原文";
    const pre = document.createElement("pre");
    pre.textContent = r.text || "";
    details.append(more, pre);
    item.append(title, loc, currency, details);
    const url = safeLink(r.source_url);
    if (url) {
      const a = document.createElement("a");
      a.href = url;
      a.target = "_blank";
      a.rel = "noopener noreferrer";
      a.textContent = "查看官方来源";
      item.append(a);
    }
    box.append(item);
  }
  return box;
}

// 读取lastResult/showAll并更新来源DOM，不联网；目录定位只在候选视图显示，不能冒充已用引用。
function sources() {
  const area = $("sources");
  if (!area) return;
  area.replaceChildren();
  
  const badge = $("aside-badge");
  
  if (!lastResult) {
    if (badge) badge.textContent = "0";
    const p = document.createElement("div");
    p.className = "empty";
    p.textContent = "本次没有可核对的已用引用，可切换查看检索候选。";
    area.append(p);
    return;
  }
  
  // 新服务返回 display 时，检索候选视图展示前8条和"更多候选"；旧结果没有 display 时照旧显示送模材料。
  const display = showAll && lastResult.display ? lastResult.display : null;
  const rows = display ? display.top || [] : showAll ? lastResult.retrieved : lastResult.citations;
  const tailRows = display ? display.tail || [] : [];
  const officialRows = showAll ? (lastResult.official_fallback || []).filter(r => r.sent_to_model === false) : [];
  
  if (badge) {
    badge.textContent = String(rows.length + officialRows.length);
  }
  
  if (!rows.length && !tailRows.length && !officialRows.length) {
    const p = document.createElement("div");
    p.className = "empty";
    p.textContent = showAll
      ? "本次没有找到可提供的资料。"
      : "本次没有可核对的已用引用，可切换查看检索候选。";
    area.append(p);
    return;
  }
  
  for (const r of rows) {
    const item = document.createElement("article");
    item.className = "source";
    
    const tag = document.createElement("span");
    tag.className = "tag";
    tag.textContent =
      (display
        ? r.sent_to_model ? (r.material_id || "") + " · 已送入模型" : "未送入模型"
        : r.material_id || r.id || "") +
      " · " +
      ((r.verified_scope || "").includes("官方白名单实时检索补充")
        ? "官方白名单实时补充 · 未核实个案适用"
        : r.content_scope === "RELEASE_SUMMARY_ONLY"
        ? "发布稿摘要 · 非判决书全文"
        : "背景资料 · 未核实个案适用");
        
    const title = document.createElement("h3");
    title.textContent = r.title;
    
    const loc = document.createElement("p");
    loc.textContent =
      (r.locator || "") + " / " + (r.version_id || "版本待核实");

    const currency = document.createElement("p");
    currency.className = "currency";
    currency.textContent = currencyNote(r.version_id);
      
    const excerpt = document.createElement("div");
    excerpt.className = "excerpt";
    excerpt.textContent =
      r.text.slice(0, 110) + (r.text.length > 110 ? "…" : "");
      
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = "展开原文";
    
    const pre = document.createElement("pre");
    pre.textContent = r.text;
    details.append(summary, pre);
    
    const note = document.createElement("p");
    note.textContent =
      "系统附列原文；" + (r.verified_scope || "核验范围待确认") + "。";
      
    item.append(tag, title, loc, currency, excerpt, details, note);
    
    const url = safeLink(r.source_url);
    if (url) {
      const a = document.createElement("a");
      a.href = url;
      a.target = "_blank";
      a.rel = "noopener noreferrer";
      a.textContent = "查看官方来源";
      item.append(a);
    }
    
    area.append(item);
  }
  
  if (showAll) {
    if (tailRows.length) area.append(tailSection(tailRows));
    for (const r of lastResult.official_fallback || []) {
      if (r.sent_to_model !== false) continue;
      const item = document.createElement("article");
      item.className = "source";
      
      const tag = document.createElement("span");
      tag.className = "tag";
      tag.textContent = "官方目录定位 · 未送入模型";
      
      const title = document.createElement("h3");
      title.textContent = r.title || "官方目录条目";
      
      const effective = document.createElement("p");
      effective.textContent = "施行日期：" + (r.effective_date || "待核实");

      const currency = document.createElement("p");
      currency.className = "currency";
      currency.textContent = "现行性：目录日期未逐条在线复核是否仍然有效；未核个案适用。";
      
      const note = document.createElement("p");
      note.textContent = "本地尚未收录全文；目录定位不构成本轮引用依据。";
      
      item.append(tag, title, effective, currency, note);
      
      const url = safeOfficialLink(r.official_url || r.url);
      if (url) {
        const a = document.createElement("a");
        a.href = url;
        a.target = "_blank";
        a.rel = "noopener noreferrer";
        a.textContent = "查看官方目录来源";
        item.append(a);
      }
      
      area.append(item);
    }
  }
}

// 刷新模型、向量入库和实际数据库后端状态
async function refresh() {
  try {
    const r = await fetch("/api/status");
    const s = await r.json();
    token = s.token;
    
    if ($("model")) {
      $("model").textContent =
        s.model +
        (s.chat_ready
          ? " · 就绪"
          : s.connected
            ? " · 尚未下载"
            : " · 服务未启动");
    }
    
    if ($("coverage")) {
      $("coverage").textContent = s.coverage.note || "资料状态待检查";
    }
    
    const idx = s.index;
    const db = s.database;
    
    if ($("database-state")) {
      $("database-state").textContent = db
        ? db.ready
          ? "Qdrant · 已同步 " + db.points + " 段 (" + db.collection + ")"
          : "Qdrant · 尚未就绪"
        : "未配置独立向量库";
    }
    
    if ($("index-state")) {
      $("index-state").textContent =
        idx.state === "ready"
          ? s.database && s.database.ready
            ? "BM25 + Qdrant 向量"
            : "关键词 + 本地向量"
          : idx.state === "building"
            ? `构建向量 ${idx.completed}/${idx.total}`
            : "关键词检索";
    }
    
    if ($("system-note")) {
      $("system-note").textContent =
        idx.notice ||
        (!s.chat_ready
          ? "请先启动 Ollama 并完成模型下载。"
          : "咨询只调用本机服务；不接入云端模型。");
    }
  } catch {
    progress("本地网页服务未连接，请重新运行启动脚本。", true);
  }
}

// 初始化页面并创建独立会话。
async function boot() {
  await refresh();
  if (token) {
    const s = await api("/api/session");
    sessionId = s.session_id;
  }
}

// 提交问题后轮询任务；epoch阻止清空会话后的旧结果重新出现在页面。
$("compose").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (busy) return;
  const question = $("question").value.trim();
  if (!question) return;
  if (!sessionId) {
    progress("连接尚未准备好，请刷新重试。", true);
    return;
  }
  
  const current = ++epoch;
  bubble("user", question);
  $("question").value = "";
  $("question").style.height = "auto";
  
  const hero = $("hero-intro");
  if (hero) hero.classList.add("collapsed");
  
  setBusy(true);
  progress("正在多路检索本地 878 部规范与 Qdrant 向量库…");
  
  try {
    const started = await api("/api/chat/start", {
      session_id: sessionId,
      question,
    });
    if (current !== epoch) return;
    requestId = started.request_id;
    
    while (current === epoch) {
      await new Promise((r) => setTimeout(r, 800));
      const status = await api("/api/chat/result", {
        session_id: sessionId,
        request_id: requestId,
      });
      if (current !== epoch) return;
      if (status.status === "running") {
        progress(status.phase || "本机大模型正在梳理事实要件与法律适用…");
        continue;
      }
      if (status.status === "done") {
        lastResult = status.result;
        const b = bubble("assistant", lastResult.answer);
        
        if (lastResult.warnings && lastResult.warnings.length) {
          const w = document.createElement("div");
          w.className = "warning";
          w.textContent =
            "⚠️ 需要复核：" +
            lastResult.warnings.map((x) => x.detail || x.type).join("；");
          b.append(w);
        }
        
        const meta = document.createElement("div");
        meta.className = "meta";
        meta.textContent = `${lastResult.elapsed}秒 · ${
          lastResult.retrieval_mode === "hybrid_qdrant"
            ? "关键词 + Qdrant 向量检索"
            : lastResult.retrieval_mode === "hybrid"
            ? "关键词与本地向量检索"
            : "关键词检索"
        } · ${lastResult.review_note}`;
        b.append(meta);
        
        if (lastResult.blocked) {
          const d = document.createElement("details");
          const s = document.createElement("summary");
          s.textContent = "查看未通过检查的原始输出";
          const pre = document.createElement("pre");
          pre.textContent = lastResult.raw_answer;
          d.append(s, pre);
          b.append(d);
        }
        
        sources();
        progress("分析已完成。已在右侧整理引用的条文原文，可点击核实。");
        showToast("法律分析完成！右侧已加载对应法条依据");
      } else if (status.status === "cancelled") {
        progress("已中止本次分析，本轮未写入会话。");
      } else {
        progress(status.error || "分析失败，请重试", true);
      }
      break;
    }
  } catch (e) {
    progress(e.message, true);
  } finally {
    if (current === epoch) {
      requestId = "";
      setBusy(false);
      $("question").focus();
    }
  }
});

// 停止请求向后端传播取消
$("stop").onclick = async () => {
  if (!requestId) return;
  try {
    await api("/api/chat/cancel", {
      session_id: sessionId,
      request_id: requestId,
    });
    progress("已发送停止请求，等待本机任务结束。");
  } catch (e) {
    progress(e.message, true);
  }
};

// 清空对话、恢复初始引导面板
$("reset").onclick = async () => {
  ++epoch;
  setBusy(true);
  try {
    const s = await api("/api/chat/reset", { session_id: sessionId });
    sessionId = s.session_id;
    requestId = "";
    lastResult = null;
    $("messages").replaceChildren();
    $("sources").replaceChildren();
    
    const hero = $("hero-intro");
    if (hero) hero.classList.remove("collapsed");
    
    const badge = $("aside-badge");
    if (badge) badge.textContent = "0";
    
    progress("已重置会话，可开始新的咨询。");
    showToast("已清空对话记录并开启新咨询");
  } catch (e) {
    progress(e.message, true);
  } finally {
    setBusy(false);
  }
};

$("rebuild").onclick = async () => {
  try {
    await api("/api/index/rebuild");
    progress("已请求重新构建本地资料向量。");
    await refresh();
    showToast("已启动资料入库与向量构建任务");
  } catch (e) {
    progress(e.message, true);
  }
};

// 常见场景快捷点击
for (const b of document.querySelectorAll("[data-question]")) {
  b.onclick = () => {
    $("question").value = b.dataset.question;
    $("question").focus();
    $("compose").requestSubmit();
  };
}

// 标签页切换：本次引用 vs 全部候选
$("used-tab").onclick = () => {
  showAll = false;
  $("used-tab").classList.add("active");
  $("all-tab").classList.remove("active");
  sources();
};

$("all-tab").onclick = () => {
  showAll = true;
  $("all-tab").classList.add("active");
  $("used-tab").classList.remove("active");
  sources();
};

// 一键全部展开/收起
let allExpanded = false;
const toggleExpandBtn = $("toggle-expand");
if (toggleExpandBtn) {
  toggleExpandBtn.onclick = () => {
    allExpanded = !allExpanded;
    toggleExpandBtn.textContent = allExpanded ? "全部收起" : "全部展开";
    const details = document.querySelectorAll("#sources details");
    for (const d of details) {
      d.open = allExpanded;
    }
  };
}

// 实时过滤条文
const filterInput = $("source-filter");
if (filterInput) {
  filterInput.addEventListener("input", (e) => {
    const term = e.target.value.trim().toLowerCase();
    const articles = document.querySelectorAll("#sources article");
    for (const a of articles) {
      const text = a.textContent.toLowerCase();
      a.hidden = term ? !text.includes(term) : false;
    }
  });
}

// 移动端依据抽屉切换
const toggleAsideBtn = $("toggle-aside");
const closeAsideBtn = $("close-aside");
const asideEl = $("reference-aside");

if (toggleAsideBtn && asideEl) {
  toggleAsideBtn.onclick = () => asideEl.classList.toggle("active");
}
if (closeAsideBtn && asideEl) {
  closeAsideBtn.onclick = () => asideEl.classList.remove("active");
}

// 输入框回车直接发送（Enter 发送，Shift+Enter 换行）
const questionInput = $("question");
questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    $("compose").requestSubmit();
  }
});

// 输入框自适应高度
questionInput.addEventListener("input", () => {
  questionInput.style.height = "auto";
  questionInput.style.height = Math.min(questionInput.scrollHeight, 180) + "px";
});

window.addEventListener("pagehide", () => {
  if (requestId)
    fetch("/api/chat/cancel", {
      method: "POST",
      keepalive: true,
      headers: { "Content-Type": "application/json", "X-Local-Token": token },
      body: JSON.stringify({ session_id: sessionId, request_id: requestId }),
    }).catch(() => {});
});

boot().catch((e) => progress(e.message, true));
setInterval(refresh, 10000);
