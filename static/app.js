'use strict';
/* ══════════════════════════════════════════════════════════
   NexusAI – Frontend  v2
   Features:
   ✅ Token-streaming (word-by-word like ChatGPT)
   ✅ Login / Register / Logout
   ✅ LocalStorage-backed chat history
   ✅ Stop generation (Escape key or button)
   ✅ Regenerate last response
   ✅ Dark / Light mode toggle
   ✅ Toast notifications
   ✅ Export chat as Markdown
   ✅ Keyboard shortcuts
   ✅ Scroll-to-bottom button
   ✅ Sidebar search
   ✅ Delete sessions with confirmation
   ✅ Drag-drop file upload
   ══════════════════════════════════════════════════════════ */

const API = '';
const STORAGE_KEY = 'nexusai_v2';

// ── App state ─────────────────────────────────────────────
let authToken   = null;   // Bearer token
let currentUser = null;   // { username, display_name, avatar }
let sessionId   = null;
let isLoading   = false;
let activeReader = null;  // current SSE ReadableStreamReader (for stop)
let lastUserMsg  = null;  // for regenerate
let sessions     = [];    // [{ id, label, messages: [{role,content}] }]
let deleteTarget = null;
let isDark       = true;

// ── DOM refs ──────────────────────────────────────────────
const $ = id => document.getElementById(id);

// Auth
const authOverlay   = $('authOverlay');
const loginView     = $('loginView');
const registerView  = $('registerView');
const loginError    = $('loginError');
const registerError = $('registerError');
const loginUsername = $('loginUsername');
const loginPassword = $('loginPassword');
const loginBtn      = $('loginBtn');
const loginEye      = $('loginEye');
const regDisplayName= $('regDisplayName');
const regUsername   = $('regUsername');
const regPassword   = $('regPassword');
const registerBtn   = $('registerBtn');

// App
const sidebar       = $('sidebar');
const sidebarToggle = $('sidebarToggle');
const sidebarClose  = $('sidebarCloseBtn');
const newChatBtn    = $('newChatBtn');
const historySearch = $('historySearch');
const historyEl     = $('chatHistory');
const messagesEl    = $('messages');
const containerEl   = $('messagesContainer');
const welcomeEl     = $('welcomeState');
const welcomeTitle  = $('welcomeTitle');
const inputEl       = $('messageInput');
const sendBtn       = $('sendBtn');
const stopBtn       = $('stopBtn');
const attachBtn     = $('attachBtn');
const fileInput     = $('fileInput');
const uploadStatus  = $('uploadStatus');
const clearChatBtn  = $('clearChatBtn');
const themeToggle   = $('themeToggle');
const scrollBtn     = $('scrollBtn');
const toastCont     = $('toastContainer');
const userAvatarSm  = $('userAvatarSm');
const userDisplayName=$('userDisplayName');
const userUsername  = $('userUsername');
const userMenuBtn   = $('userMenuBtn');
const userDropdown  = $('userDropdown');
const logoutBtn     = $('logoutBtn');
const exportChatBtn = $('exportChatBtn');
const shortcutsBtn  = $('shortcutsBtn');
const shortcutsModal= $('shortcutsModal');
const closeShortcuts= $('closeShortcuts');
const deleteModal   = $('deleteModal');
const deleteModalLabel=$('deleteModalLabel');
const deleteCancelBtn=$('deleteCancelBtn');
const deleteConfirmBtn=$('deleteConfirmBtn');
const settingsBtn   = $('settingsBtn');
const settingsModal = $('settingsModal');
const closeSettings = $('closeSettings');
const settingsCancelBtn = $('settingsCancelBtn');
const settingsSaveBtn = $('settingsSaveBtn');
const themeSelect   = $('themeSelect');
const temperatureSlider = $('temperatureSlider');
const tempValue     = $('tempValue');
const systemPrompt  = $('systemPrompt');

// ══════════════════════════════════════════════════════════
//  MARKED / HLJS CONFIG
// ══════════════════════════════════════════════════════════
const renderer = new marked.Renderer();

renderer.code = (code, lang) => {
  let highlighted;
  try {
    highlighted = lang && hljs.getLanguage(lang)
      ? hljs.highlight(code, { language: lang }).value
      : hljs.highlightAuto(code).value;
  } catch (_) { highlighted = escHtml(code); }
  const l = lang || 'text';
  return `<pre>
<div class="code-header">
  <span>${escHtml(l)}</span>
  <button class="copy-btn" onclick="copyCode(this)">
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
    Copy
  </button>
</div>
<code class="hljs language-${escHtml(l)}">${highlighted}</code></pre>`;
};

marked.use({ renderer, breaks: true, gfm: true });

// ══════════════════════════════════════════════════════════
//  HELPERS
// ══════════════════════════════════════════════════════════
function escHtml(s) {
  return String(s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function renderMd(text) { return marked.parse(text); }

function scrollToBottom(smooth = true) {
  containerEl.scrollTo({ top: containerEl.scrollHeight, behavior: smooth ? 'smooth' : 'instant' });
}

function autoResize() {
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 220) + 'px';
}

function showMessages() {
  welcomeEl.style.display = 'none';
  messagesEl.style.display = 'flex';
}

function showWelcome() {
  welcomeEl.style.display = 'flex';
  messagesEl.style.display = 'none';
  messagesEl.innerHTML = '';
}

// ── Toast ─────────────────────────────────────────────────
function toast(msg, type = 'success') {
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.innerHTML = type === 'success'
    ? `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#10a37f" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg> ${escHtml(msg)}`
    : `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#ef4444" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg> ${escHtml(msg)}`;
  toastCont.appendChild(el);
  setTimeout(() => el.remove(), 3200);
}

// ── Copy code from code block ─────────────────────────────
window.copyCode = function(btn) {
  const code = btn.closest('pre').querySelector('code');
  navigator.clipboard.writeText(code.innerText).then(() => {
    btn.classList.add('copied');
    btn.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#10a37f" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg> Copied!`;
    setTimeout(() => {
      btn.classList.remove('copied');
      btn.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg> Copy`;
    }, 2000);
  });
};

// ══════════════════════════════════════════════════════════
//  LOCALSTORAGE
// ══════════════════════════════════════════════════════════
function saveData() {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify({ sessions, theme: isDark ? 'dark' : 'light' })); } catch (_) {}
}

function loadData() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    const data = JSON.parse(raw);
    sessions = data.sessions || [];
    if (data.theme) { isDark = data.theme === 'dark'; applyTheme(); }
  } catch (_) { sessions = []; }
}

function getSession(id) { return sessions.find(s => s.id === id) || null; }

// ══════════════════════════════════════════════════════════
//  THEME
// ══════════════════════════════════════════════════════════
function applyTheme() {
  document.documentElement.setAttribute('data-theme', isDark ? 'dark' : 'light');
  const sunIcon = themeToggle.querySelector('.sun-icon');
  const moonIcon = themeToggle.querySelector('.moon-icon');
  sunIcon.style.display = isDark ? 'block' : 'none';
  moonIcon.style.display = isDark ? 'none' : 'block';
  // Switch highlight.js theme
  const link = document.getElementById('hljs-theme');
  if (link) link.href = isDark
    ? 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css'
    : 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css';
}

themeToggle.addEventListener('click', () => {
  isDark = !isDark;
  applyTheme();
  saveData();
  toast(isDark ? 'Dark mode on' : 'Light mode on');
});

// ══════════════════════════════════════════════════════════
//  AUTH
// ══════════════════════════════════════════════════════════
function authHeaders() {
  return { 'Content-Type': 'application/json', 'Authorization': `Bearer ${authToken}` };
}

function showAuthOverlay() { authOverlay.classList.remove('hidden'); }
function hideAuthOverlay() { authOverlay.classList.add('hidden'); }

function setUserUI(user) {
  currentUser = user;
  userAvatarSm.textContent = user.avatar || user.display_name[0].toUpperCase();
  userDisplayName.textContent = user.display_name;
  userUsername.textContent = `@${user.username}`;
  if (welcomeTitle) welcomeTitle.textContent = `How can I help you, ${user.display_name.split(' ')[0]}?`;
}

// Toggle login/register views
$('goRegister').onclick = () => { loginView.style.display='none'; registerView.style.display='block'; loginError.classList.remove('show'); };
$('goLogin').onclick = () => { registerView.style.display='none'; loginView.style.display='block'; registerError.classList.remove('show'); };

// Password eye toggle
loginEye.onclick = () => {
  const t = loginPassword.type === 'password' ? 'text' : 'password';
  loginPassword.type = t;
  loginEye.querySelector('.show-eye').style.display = t === 'password' ? 'block' : 'none';
  loginEye.querySelector('.hide-eye').style.display = t === 'text' ? 'block' : 'none';
};

// ── Login ──────────────────────────────────────────────────
loginBtn.onclick = handleLogin;
[loginUsername, loginPassword].forEach(el => el.addEventListener('keydown', e => { if (e.key === 'Enter') handleLogin(); }));

async function handleLogin() {
  const username = loginUsername.value.trim();
  const password = loginPassword.value;
  if (!username || !password) { showAuthError(loginError, 'Enter username and password'); return; }
  loginBtn.disabled = true;
  loginBtn.textContent = 'Signing in…';
  try {
    const res = await fetch(`${API}/api/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    });
    const data = await res.json();
    if (!res.ok) { showAuthError(loginError, data.detail || 'Login failed'); return; }
    authToken = data.token;
    localStorage.setItem('nexusai_token', authToken);
    setUserUI(data);
    hideAuthOverlay();
    loadData();
    renderHistory();
    toast(`Welcome back, ${data.display_name}! 👋`);
  } catch (e) {
    showAuthError(loginError, 'Connection error. Is the server running?');
  } finally {
    loginBtn.disabled = false;
    loginBtn.textContent = 'Sign in';
  }
}

// ── Register ───────────────────────────────────────────────
registerBtn.onclick = handleRegister;

async function handleRegister() {
  const username = regUsername.value.trim().toLowerCase();
  const password = regPassword.value;
  const displayName = regDisplayName.value.trim();
  if (!username || !password) { showAuthError(registerError, 'Fill in all fields'); return; }
  registerBtn.disabled = true;
  registerBtn.textContent = 'Creating account…';
  try {
    const res = await fetch(`${API}/api/auth/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password, display_name: displayName }),
    });
    const data = await res.json();
    if (!res.ok) { showAuthError(registerError, data.detail || 'Registration failed'); return; }
    toast('Account created! Sign in now.');
    registerView.style.display = 'none';
    loginView.style.display = 'block';
    loginUsername.value = username;
  } catch (e) {
    showAuthError(registerError, 'Connection error.');
  } finally {
    registerBtn.disabled = false;
    registerBtn.textContent = 'Create account';
  }
}

function showAuthError(el, msg) {
  el.textContent = msg;
  el.classList.add('show');
  setTimeout(() => el.classList.remove('show'), 5000);
}

// ── Logout ─────────────────────────────────────────────────
logoutBtn.onclick = async () => {
  try { await fetch(`${API}/api/auth/logout`, { method: 'POST', headers: authHeaders() }); } catch (_) {}
  authToken = null; currentUser = null;
  localStorage.removeItem('nexusai_token');
  sessionId = null; sessions = [];
  showWelcome(); renderHistory();
  userDropdown.style.display = 'none';
  loginView.style.display = 'block';
  registerView.style.display = 'none';
  loginUsername.value = ''; loginPassword.value = '';
  loginError.classList.remove('show');
  showAuthOverlay();
  toast('Logged out successfully');
};

// ── Check existing token on load ───────────────────────────
async function checkAuth() {
  const saved = localStorage.getItem('nexusai_token');
  if (!saved) { showAuthOverlay(); return false; }
  authToken = saved;
  try {
    const res = await fetch(`${API}/api/auth/me`, { headers: authHeaders() });
    if (!res.ok) { localStorage.removeItem('nexusai_token'); authToken = null; showAuthOverlay(); return false; }
    const data = await res.json();
    setUserUI(data);
    hideAuthOverlay();
    return true;
  } catch (_) { showAuthOverlay(); return false; }
}

// ══════════════════════════════════════════════════════════
//  USER DROPDOWN
// ══════════════════════════════════════════════════════════
userMenuBtn.onclick = (e) => {
  e.stopPropagation();
  const isOpen = userDropdown.style.display === 'block';
  userDropdown.style.display = isOpen ? 'none' : 'block';
};
document.addEventListener('click', () => { userDropdown.style.display = 'none'; });

// ══════════════════════════════════════════════════════════
//  HISTORY / SESSIONS
// ══════════════════════════════════════════════════════════
function renderHistory(filter = '') {
  historyEl.innerHTML = '';
  const q = filter.toLowerCase().trim();
  const visible = [...sessions].reverse().filter(s => !q || s.label.toLowerCase().includes(q));
  if (!visible.length) {
    historyEl.innerHTML = `<div style="font-size:13px;color:var(--text-f);padding:10px 8px">${q ? 'No chats match' : 'No chats yet'}</div>`;
    return;
  }
  visible.forEach(s => {
    const item = document.createElement('div');
    item.className = 'history-item' + (s.id === sessionId ? ' active' : '');

    const label = document.createElement('span');
    label.className = 'history-item-label';
    label.textContent = s.label;

    const acts = document.createElement('div');
    acts.className = 'history-item-actions';
    const delBtn = document.createElement('button');
    delBtn.title = 'Delete';
    delBtn.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14H6L5 6"/><path d="M10 11v6M14 11v6"/><path d="M9 6V4h6v2"/></svg>`;
    delBtn.onclick = e => { e.stopPropagation(); openDeleteModal(s.id, s.label); };
    acts.appendChild(delBtn);
    item.appendChild(label); item.appendChild(acts);
    item.onclick = () => switchSession(s.id);
    historyEl.appendChild(item);
  });
}

historySearch.addEventListener('input', () => renderHistory(historySearch.value));

function switchSession(id) {
  sessionId = id;
  renderHistory(historySearch.value);
  const sess = getSession(id);
  if (!sess) return;
  messagesEl.innerHTML = '';
  if (sess.messages && sess.messages.length) {
    showMessages();
    sess.messages.forEach(m => appendMessage(m.role, m.content, false));
    scrollToBottom(false);
    // Show regenerate on last AI message
    attachRegenToLast();
  } else { showWelcome(); }
}

function attachRegenToLast() {
  const allAssistant = messagesEl.querySelectorAll('.message.assistant');
  const last = allAssistant[allAssistant.length - 1];
  if (last) appendRegenBtn(last);
}

async function createSession() {
  const id = 'local-' + crypto.randomUUID();
  try {
    const res = await fetch(`${API}/api/session`, { method: 'POST', headers: authHeaders() });
    if (res.ok) { const d = await res.json(); sessions.push({ id: d.session_id, label: 'New chat', messages: [] }); sessionId = d.session_id; }
    else { sessions.push({ id, label: 'New chat', messages: [] }); sessionId = id; }
  } catch (_) { sessions.push({ id, label: 'New chat', messages: [] }); sessionId = id; }
  saveData(); renderHistory(historySearch.value);
}

// ── Delete modal ──────────────────────────────────────────
function openDeleteModal(id, label) { deleteTarget = id; deleteModalLabel.textContent = `"${label}"`; deleteModal.style.display = 'flex'; }
deleteCancelBtn.onclick = () => { deleteModal.style.display = 'none'; deleteTarget = null; };
deleteConfirmBtn.onclick = async () => {
  if (!deleteTarget) return;
  try { await fetch(`${API}/api/session/${deleteTarget}`, { method: 'DELETE', headers: authHeaders() }); } catch (_) {}
  sessions = sessions.filter(s => s.id !== deleteTarget);
  saveData();
  if (sessionId === deleteTarget) { sessionId = null; showWelcome(); }
  deleteModal.style.display = 'none'; deleteTarget = null;
  renderHistory(historySearch.value);
  toast('Chat deleted');
};
deleteModal.addEventListener('click', e => { if (e.target === deleteModal) { deleteModal.style.display = 'none'; deleteTarget = null; } });

// ══════════════════════════════════════════════════════════
//  MESSAGE RENDERING
// ══════════════════════════════════════════════════════════
const AI_AVATAR_SVG = `<svg width="16" height="16" viewBox="0 0 41 41" fill="none"><path d="M37.532 16.87a9.963 9.963 0 0 0-.856-8.184 10.078 10.078 0 0 0-10.855-4.835 9.964 9.964 0 0 0-6.75-3.668 10.078 10.078 0 0 0-10.855 4.835 9.965 9.965 0 0 0-6.75 3.668 10.079 10.079 0 0 0-1.252 11.954 9.963 9.963 0 0 0 .856 8.185 10.079 10.079 0 0 0 10.855 4.835 9.965 9.965 0 0 0 6.75 3.668 10.079 10.079 0 0 0 10.855-4.835 9.965 9.965 0 0 0 6.75-3.668 10.079 10.079 0 0 0 1.252-11.954Z" fill="currentColor"/></svg>`;

function appendMessage(role, content, save = true) {
  showMessages();
  const wrap = document.createElement('div');
  wrap.className = `message ${role}`;
  const inner = document.createElement('div');
  inner.className = 'message-inner';

  const avatar = document.createElement('div');
  avatar.className = `avatar ${role === 'user' ? 'user-avatar' : 'assistant-avatar'}`;
  avatar.innerHTML = role === 'user'
    ? (currentUser ? currentUser.avatar : 'U')
    : AI_AVATAR_SVG;

  if (role === 'user') {
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.innerHTML = renderMd(escHtml(content));
    inner.appendChild(avatar); inner.appendChild(bubble);
    wrap.appendChild(inner);
  } else {
    const col = document.createElement('div');
    col.style.flex = '1'; col.style.minWidth = '0';
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.innerHTML = renderMd(content);
    // Message action buttons
    const acts = buildMsgActions(bubble, content);
    col.appendChild(bubble); col.appendChild(acts);
    inner.appendChild(avatar); inner.appendChild(col);
    wrap.appendChild(inner);
  }

  messagesEl.appendChild(wrap);
  scrollToBottom();

  if (save && sessionId) {
    const sess = getSession(sessionId);
    if (sess) { sess.messages.push({ role, content }); saveData(); }
  }
  return wrap;
}

function buildMsgActions(bubble, content) {
  const acts = document.createElement('div');
  acts.className = 'message-actions';
  acts.innerHTML = `
    <button class="msg-btn" title="Copy response">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
    </button>
    <button class="msg-btn" title="Good response">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3H14z"/><path d="M7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"/></svg>
    </button>
    <button class="msg-btn" title="Bad response">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3H10z"/><path d="M17 2h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17"/></svg>
    </button>`;
  acts.children[0].onclick = () => {
    navigator.clipboard.writeText(bubble.innerText);
    acts.children[0].classList.add('active');
    setTimeout(() => acts.children[0].classList.remove('active'), 1500);
    toast('Copied to clipboard');
  };
  return acts;
}

function appendRegenBtn(msgEl) {
  // Remove existing regen btn
  msgEl.querySelectorAll('.regen-btn').forEach(b => b.remove());
  const btn = document.createElement('button');
  btn.className = 'regen-btn';
  btn.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg> Regenerate response`;
  btn.onclick = () => regenerate();
  const col = msgEl.querySelector('.message-inner > div:last-child') || msgEl.querySelector('.message-inner');
  col.appendChild(btn);
}

// ══════════════════════════════════════════════════════════
//  STREAMING
// ══════════════════════════════════════════════════════════
function setLoading(loading) {
  isLoading = loading;
  inputEl.disabled = loading;
  sendBtn.style.display = loading ? 'none' : 'flex';
  stopBtn.style.display = loading ? 'flex' : 'none';
  if (!loading) sendBtn.disabled = !inputEl.value.trim();
}

async function sendMessage(text) {
  text = text.trim();
  if (isLoading || !text || !authToken) return;

  if (!sessionId) await createSession();
  lastUserMsg = text;

  setLoading(true);
  appendMessage('user', text);
  inputEl.value = ''; autoResize();

  // Show typing dots then replace with streaming bubble
  const typingEl = showTypingIndicator();

  // Build streaming assistant bubble (hidden initially)
  const wrap = document.createElement('div');
  wrap.className = 'message assistant';
  const inner = document.createElement('div'); inner.className = 'message-inner';
  const avatar = document.createElement('div'); avatar.className = 'avatar assistant-avatar'; avatar.innerHTML = AI_AVATAR_SVG;
  const col = document.createElement('div'); col.style.flex = '1'; col.style.minWidth = '0';
  const bubble = document.createElement('div'); bubble.className = 'bubble stream-cursor';
  col.appendChild(bubble); inner.appendChild(avatar); inner.appendChild(col); wrap.appendChild(inner);

  let fullText = '';
  let streamStarted = false;

  try {
    const res = await fetch(`${API}/api/chat/stream`, {
      method: 'POST',
      headers: authHeaders(),
      body: JSON.stringify({ session_id: sessionId, message: text }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Error ${res.status}`);
    }

    const reader = res.body.getReader();
    activeReader = reader;
    const decoder = new TextDecoder();
    let buf = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split('\n');
      buf = lines.pop();

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        let data;
        try { data = JSON.parse(line.slice(6)); } catch (_) { continue; }

        if (data.token) {
          if (!streamStarted) {
            typingEl.remove();
            messagesEl.appendChild(wrap);
            streamStarted = true;
          }
          fullText += data.token;
          // Progressive markdown render
          bubble.innerHTML = renderMd(fullText);
          bubble.classList.add('stream-cursor');
          scrollToBottom();
        }

        if (data.session_id && !sessionId.startsWith('local-')) {
          sessionId = data.session_id;
        }

        if (data.done) {
          // Finalize
          bubble.classList.remove('stream-cursor');
          bubble.innerHTML = renderMd(fullText);

          // Save to local store
          const sess = getSession(sessionId);
          if (sess) {
            sess.messages.push({ role: 'assistant', content: fullText });
            if (sess.label === 'New chat') {
              sess.label = text.slice(0, 42) + (text.length > 42 ? '…' : '');
            }
            saveData();
            renderHistory(historySearch.value);
          }

          // Add action buttons + regenerate
          const acts = buildMsgActions(bubble, fullText);
          col.appendChild(acts);
          appendRegenBtn(wrap);
          scrollToBottom();
        }

        if (data.error) throw new Error(data.error);
      }
    }
  } catch (err) {
    if (err.name === 'AbortError' || err.message === 'stop') {
      // Stopped by user
      bubble.classList.remove('stream-cursor');
      if (fullText) bubble.innerHTML = renderMd(fullText + '\n\n*[Generation stopped]*');
      toast('Generation stopped');
    } else {
      typingEl.remove();
      appendMessage('assistant', `⚠️ **Error:** ${err.message}`);
    }
  } finally {
    activeReader = null;
    if (!streamStarted && typingEl.parentNode) typingEl.remove();
    setLoading(false);
    inputEl.focus();
  }
}

function showTypingIndicator() {
  const tpl = document.getElementById('typingTpl').content.cloneNode(true);
  const el = tpl.querySelector('.typing-msg');
  messagesEl.appendChild(el);
  scrollToBottom();
  return messagesEl.lastElementChild;
}

// ── Stop generation ───────────────────────────────────────
stopBtn.onclick = stopGeneration;

function stopGeneration() {
  if (activeReader) {
    activeReader.cancel('stop').catch(() => {});
    activeReader = null;
  }
}

// ── Regenerate ────────────────────────────────────────────
async function regenerate() {
  if (isLoading || !lastUserMsg || !sessionId) return;
  // Remove last assistant message from DOM and store
  const allAssistant = messagesEl.querySelectorAll('.message.assistant');
  const lastMsg = messagesEl.querySelectorAll('.message.user');
  const lastAI = allAssistant[allAssistant.length - 1];
  if (lastAI) lastAI.remove();

  // Remove from history store
  const sess = getSession(sessionId);
  if (sess && sess.messages.length > 0 && sess.messages[sess.messages.length - 1].role === 'assistant') {
    sess.messages.pop();
    saveData();
  }

  // Resend
  await sendMessage(lastUserMsg);
}

// ══════════════════════════════════════════════════════════
//  INPUT EVENTS
// ══════════════════════════════════════════════════════════
inputEl.addEventListener('input', () => {
  autoResize();
  sendBtn.disabled = !inputEl.value.trim() || isLoading;
});

inputEl.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    if (!sendBtn.disabled) sendMessage(inputEl.value);
  }
});

sendBtn.addEventListener('click', () => sendMessage(inputEl.value));

window.insertSuggestion = function(text) {
  inputEl.value = text; autoResize(); sendBtn.disabled = false;
  sendMessage(text);
};

// ══════════════════════════════════════════════════════════
//  SIDEBAR
// ══════════════════════════════════════════════════════════
let sidebarOpen = true;

function toggleSidebar() {
  const mobile = window.innerWidth <= 768;
  if (mobile) { sidebar.classList.toggle('mobile-open'); }
  else { sidebarOpen = !sidebarOpen; sidebar.classList.toggle('collapsed', !sidebarOpen); }
}

sidebarToggle.addEventListener('click', toggleSidebar);
sidebarClose.addEventListener('click', toggleSidebar);
document.addEventListener('click', e => {
  if (window.innerWidth <= 768 && sidebar.classList.contains('mobile-open')) {
    if (!sidebar.contains(e.target) && e.target !== sidebarToggle) sidebar.classList.remove('mobile-open');
  }
});

// ── New chat ──────────────────────────────────────────────
newChatBtn.onclick = () => { sessionId = null; showWelcome(); renderHistory(historySearch.value); inputEl.focus(); };

// ── Clear chat ────────────────────────────────────────────
clearChatBtn.onclick = async () => {
  if (!sessionId) return;
  try { await fetch(`${API}/api/session/${sessionId}`, { method: 'DELETE', headers: authHeaders() }); } catch (_) {}
  const sess = getSession(sessionId);
  if (sess) { sess.messages = []; saveData(); }
  sessionId = null; showWelcome(); renderHistory(historySearch.value);
  toast('Chat cleared');
};

// ══════════════════════════════════════════════════════════
//  SCROLL TO BOTTOM BUTTON
// ══════════════════════════════════════════════════════════
containerEl.addEventListener('scroll', () => {
  const atBottom = containerEl.scrollHeight - containerEl.scrollTop - containerEl.clientHeight < 80;
  scrollBtn.style.display = atBottom ? 'none' : 'flex';
});
scrollBtn.addEventListener('click', () => scrollToBottom());

// ══════════════════════════════════════════════════════════
//  FILE UPLOAD
// ══════════════════════════════════════════════════════════
attachBtn.addEventListener('click', () => fileInput.click());
uploadArea.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => { if (fileInput.files[0]) handleUpload(fileInput.files[0]); });

// Drag & drop on input wrapper
const inputWrapper = $('inputWrapper');
inputWrapper.addEventListener('dragover', e => { e.preventDefault(); inputWrapper.style.borderColor = 'var(--accent)'; });
inputWrapper.addEventListener('dragleave', () => { inputWrapper.style.borderColor = ''; });
inputWrapper.addEventListener('drop', e => {
  e.preventDefault(); inputWrapper.style.borderColor = '';
  if (e.dataTransfer.files[0]) handleUpload(e.dataTransfer.files[0]);
});

async function handleUpload(file) {
  if (!authToken) { toast('Please log in first', 'error'); return; }
  if (!sessionId) await createSession();
  uploadStatus.textContent = `Uploading ${file.name}…`;
  const fd = new FormData(); fd.append('file', file); fd.append('session_id', sessionId);
  try {
    const res = await fetch(`${API}/api/upload`, { method: 'POST', headers: { 'Authorization': `Bearer ${authToken}` }, body: fd });
    const data = await res.json();
    if (data.status?.includes('error')) { uploadStatus.textContent = '✗ Failed'; toast(data.message || 'Upload failed', 'error'); }
    else { uploadStatus.textContent = `✓ ${data.filename}`; toast(`Indexed ${data.chunks} chunks from ${data.filename}`); }
  } catch (_) { uploadStatus.textContent = '✗ Upload failed'; toast('Upload failed', 'error'); }
  setTimeout(() => { uploadStatus.textContent = ''; }, 4000);
}

// ══════════════════════════════════════════════════════════
//  EXPORT CHAT
// ══════════════════════════════════════════════════════════
exportChatBtn.onclick = () => {
  userDropdown.style.display = 'none';
  const sess = getSession(sessionId);
  if (!sess || !sess.messages.length) { toast('No messages to export', 'error'); return; }
  let md = `# ${sess.label}\n\n`;
  sess.messages.forEach(m => { md += `**${m.role === 'user' ? '👤 You' : '🤖 NexusAI'}:**\n${m.content}\n\n---\n\n`; });
  const blob = new Blob([md], { type: 'text/markdown' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `${sess.label.replace(/[^a-z0-9]/gi, '-')}.md`;
  a.click(); URL.revokeObjectURL(a.href);
  toast('Chat exported as Markdown');
};

// ══════════════════════════════════════════════════════════
//  KEYBOARD SHORTCUTS
// ══════════════════════════════════════════════════════════
shortcutsBtn.onclick = () => { userDropdown.style.display = 'none'; shortcutsModal.style.display = 'flex'; };
closeShortcuts.onclick = () => { shortcutsModal.style.display = 'none'; };
shortcutsModal.addEventListener('click', e => { if (e.target === shortcutsModal) shortcutsModal.style.display = 'none'; });

document.addEventListener('keydown', e => {
  const ctrl = e.ctrlKey || e.metaKey;
  if (e.key === 'Escape') { stopGeneration(); shortcutsModal.style.display = 'none'; deleteModal.style.display = 'none'; }
  if (ctrl && e.key === 'n') { e.preventDefault(); newChatBtn.click(); }
  if (ctrl && e.key === 'b') { e.preventDefault(); toggleSidebar(); }
  if (ctrl && e.key === 'l') { e.preventDefault(); if (sessionId) clearChatBtn.click(); }
  if (ctrl && e.key === 'e') { e.preventDefault(); exportChatBtn.click(); }
});

// ══════════════════════════════════════════════════════════
//  SETTINGS
// ══════════════════════════════════════════════════════════
settingsBtn.addEventListener('click', () => {
  userDropdown.style.display = 'none';
  themeSelect.value = isDark ? 'dark' : 'light';
  settingsModal.style.display = 'flex';
});
closeSettings.addEventListener('click', () => settingsModal.style.display = 'none');
settingsCancelBtn.addEventListener('click', () => settingsModal.style.display = 'none');
settingsModal.addEventListener('click', e => { if (e.target === settingsModal) settingsModal.style.display = 'none'; });

temperatureSlider.addEventListener('input', (e) => tempValue.textContent = e.target.value);

settingsSaveBtn.addEventListener('click', () => {
  const newTheme = themeSelect.value;
  if (newTheme !== (isDark ? 'dark' : 'light')) {
    isDark = newTheme === 'dark';
    applyTheme();
    saveData();
  }
  
  localStorage.setItem('nexusai_temperature', temperatureSlider.value);
  localStorage.setItem('nexusai_system_prompt', systemPrompt.value);
  
  toast('Settings saved successfully');
  settingsModal.style.display = 'none';
});

// Restore settings on load
document.addEventListener('DOMContentLoaded', () => {
  const t = localStorage.getItem('nexusai_temperature');
  if (t) {
    temperatureSlider.value = t;
    tempValue.textContent = t;
  }
  const sp = localStorage.getItem('nexusai_system_prompt');
  if (sp) {
    systemPrompt.value = sp;
  }
});

// ══════════════════════════════════════════════════════════
//  INIT
// ══════════════════════════════════════════════════════════
async function init() {
  loadData();
  applyTheme();
  const authed = await checkAuth();
  if (authed) { renderHistory(); showWelcome(); }
  inputEl.focus();
  if (window.innerWidth <= 768) { sidebarOpen = false; }
}

init();
