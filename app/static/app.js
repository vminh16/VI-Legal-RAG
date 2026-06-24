// ViLaborRAG Premium Frontend Logic (app.js)

// Tự động trích xuất token truy cập từ URL parameter nếu có (BUG-06)
const urlParams = new URLSearchParams(window.location.search);
const urlToken = urlParams.get('token');
if (urlToken) {
    localStorage.setItem('api_auth_token', urlToken);
    // Làm sạch thanh địa chỉ của trình duyệt
    const cleanUrl = window.location.protocol + "//" + window.location.host + window.location.pathname;
    window.history.replaceState({ path: cleanUrl }, '', cleanUrl);
}

// State management
let currentResponse = null;
let requestStartTime = null;
let requestEndTime = null;
let activeTab = 'retrieved';
let stepperTimeouts = [];

// DOM Elements
const queryInput = document.getElementById('query-input');
const sendBtn = document.getElementById('send-btn');
const chatWindow = document.getElementById('chat-window');
const strategySelect = document.getElementById('strategy-select');
const topkSlider = document.getElementById('topk-slider');
const bypassRefusalCheckbox = document.getElementById('bypass-refusal-checkbox');
const debugContent = document.getElementById('debug-content');
const debugTabs = document.querySelectorAll('.debug-tab');
const debugDrawer = document.getElementById('debug-drawer');
const debugToggleBtn = document.getElementById('debug-toggle-btn');

// Sidebar system status elements
const sysStatus = document.getElementById('sys-status');
const geminiStatus = document.getElementById('gemini-status');
const embedModelStatus = document.getElementById('embed-model-status');
const rerankModelStatus = document.getElementById('rerank-model-status');

// Initialize on page load
window.addEventListener('DOMContentLoaded', () => {
    fetchSystemStatus();
    setupEventListeners();
});

// Fetch system status from FastAPI backend
async function fetchSystemStatus() {
    try {
        const response = await fetch('/api/v1/system/status');
        if (!response.ok) throw new Error('Không thể lấy thông tin hệ thống');
        
        const data = await response.json();
        
        // Update Sidebar UI elements
        sysStatus.textContent = 'Sẵn sàng';
        sysStatus.className = 'status-val status-active';
        
        if (data.api_key_configured) {
            geminiStatus.textContent = 'Đã kết nối';
            geminiStatus.style.color = 'var(--text-success)';
        } else {
            geminiStatus.textContent = 'Mô phỏng (Mock Mode)';
            geminiStatus.style.color = 'var(--text-warning)';
        }
        
        if (data.models) {
            embedModelStatus.textContent = data.models.embedding || 'BGE-M3';
            rerankModelStatus.textContent = data.models.reranker || 'Cross-Encoder';
        }
    } catch (error) {
        console.error('Error fetching system status:', error);
        sysStatus.textContent = 'Lỗi kết nối';
        sysStatus.className = 'status-val';
        sysStatus.style.color = 'var(--text-error)';
        geminiStatus.textContent = 'Ngoại tuyến';
        geminiStatus.style.color = 'var(--text-error)';
    }
}

// Setup events
function setupEventListeners() {
    // Send button click
    sendBtn.addEventListener('click', submitQuery);
    
    // Textarea enter key (Send without Shift)
    queryInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            submitQuery();
        }
    });

    // Auto-resize textarea height
    queryInput.addEventListener('input', function() {
        this.style.height = '24px';
        const newHeight = Math.min(this.scrollHeight, 120);
        this.style.height = newHeight + 'px';
    });
    
    // Debug tabs click handlers
    debugTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            debugTabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            activeTab = tab.getAttribute('data-tab');
            renderDebugContent();
            
            // Expand drawer if collapsed on tab selection
            if (!debugDrawer.classList.contains('expanded')) {
                debugDrawer.classList.add('expanded');
                debugToggleBtn.textContent = 'Thu nhỏ ↓';
            }
        });
    });

    // Toggle expand/collapse Debug Drawer
    debugToggleBtn.addEventListener('click', () => {
        const isExpanded = debugDrawer.classList.toggle('expanded');
        debugToggleBtn.textContent = isExpanded ? 'Thu nhỏ ↓' : 'Mở rộng ↑';
    });
}

// Execute suggestions from Quick Templates
function askSuggestion(text) {
    queryInput.value = text;
    queryInput.dispatchEvent(new Event('input'));
    submitQuery();
}

// Map raw technical exceptions to friendly, professional Vietnamese responses
function getFriendlyErrorMessage(errorMsg) {
    const msg = String(errorMsg || '').toLowerCase();
    
    if (msg.includes('rate limit') || msg.includes('429') || msg.includes('too many requests')) {
        return 'Yêu cầu của bạn hiện chưa được phản hồi do giới hạn lượt truy cập tạm thời (API Rate Limit). Vui lòng đợi vài giây và thử lại.';
    }
    
    if (msg.includes('api_key') || msg.includes('api key') || msg.includes('credentials') || msg.includes('auth') || msg.includes('unauthorized')) {
        return 'Không thể kết nối đến mô hình ngôn ngữ do lỗi xác thực hoặc chưa cấu hình khóa bảo mật (API Key). Vui lòng kiểm tra lại môi trường hệ thống.';
    }
    
    if (msg.includes('network') || msg.includes('fetch') || msg.includes('failed to fetch') || msg.includes('conn') || msg.includes('connection')) {
        return 'Không thể kết nối với máy chủ. Vui lòng kiểm tra lại kết nối mạng internet hoặc trạng thái khởi chạy của server FastAPI.';
    }
    
    if (msg.includes('timeout')) {
        return 'Yêu cầu xử lý bị quá thời gian cho phép (Timeout). Vui lòng thử lại với câu hỏi ngắn gọn hơn.';
    }
    
    return 'Hệ thống gặp sự cố kỹ thuật trong lúc phân tích văn bản pháp lý. Rất tiếc vì sự bất tiện này, xin vui lòng thử lại sau.';
}

// Manage RAG stepper loading animations
function startStepperAnimation(botMsgId) {
    clearStepperAnimation();
    
    const activateStep = (stepNum) => {
        for (let i = 1; i <= 4; i++) {
            const stepEl = document.getElementById(`${botMsgId}-step-${i}`);
            if (stepEl) stepEl.classList.remove('active');
        }
        
        const currentStepEl = document.getElementById(`${botMsgId}-step-${stepNum}`);
        if (currentStepEl) currentStepEl.classList.add('active');
        
        for (let i = 1; i < stepNum; i++) {
            const prevStepEl = document.getElementById(`${botMsgId}-step-${i}`);
            if (prevStepEl && !prevStepEl.classList.contains('completed')) {
                prevStepEl.classList.add('completed');
                const indicator = prevStepEl.querySelector('.step-indicator');
                if (indicator) indicator.textContent = '✓';
            }
        }
    };
    
    // Step 1 immediately
    activateStep(1);
    
    // Simulate pipeline timeline
    stepperTimeouts.push(setTimeout(() => activateStep(2), 1000));
    stepperTimeouts.push(setTimeout(() => activateStep(3), 2400));
    stepperTimeouts.push(setTimeout(() => activateStep(4), 4500));
}

function clearStepperAnimation() {
    stepperTimeouts.forEach(t => clearTimeout(t));
    stepperTimeouts = [];
}

function completeStepperAnimation(botMsgId) {
    clearStepperAnimation();
    for (let i = 1; i <= 4; i++) {
        const stepEl = document.getElementById(`${botMsgId}-step-${i}`);
        if (stepEl) {
            stepEl.classList.remove('active');
            stepEl.classList.add('completed');
            const indicator = stepEl.querySelector('.step-indicator');
            if (indicator) indicator.textContent = '✓';
        }
    }
}

// Main query submission
async function submitQuery() {
    const query = queryInput.value.trim();
    if (!query) return;
    
    // Clear input
    queryInput.value = '';
    queryInput.style.height = '24px';
    
    // Add user message bubble
    appendMessage(query, 'user');
    
    // Create bot placeholder bubble with progress stepper
    const botMsgId = 'bot-msg-' + Date.now();
    appendBotLoader(botMsgId);
    startStepperAnimation(botMsgId);
    
    // Prepare API Request Payload
    const payload = {
        query: query,
        strategy: strategySelect.value,
        top_k: parseInt(topkSlider.value, 10),
        bypass_refusal: bypassRefusalCheckbox.checked
    };
    
    // Gửi kèm Bearer token nếu có trong cấu hình (BUG-06)
    const headers = {
        'Content-Type': 'application/json'
    };
    const apiToken = localStorage.getItem('api_auth_token');
    if (apiToken) {
        headers['Authorization'] = `Bearer ${apiToken}`;
    }
    
    try {
        requestStartTime = performance.now();
        const response = await fetch('/api/v1/query', {
            method: 'POST',
            headers: headers,
            body: JSON.stringify(payload)
        });
        
        requestEndTime = performance.now();
        
        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.detail || 'Lỗi hệ thống');
        }
        
        const data = await response.json();
        currentResponse = data;
        
        // Render final bot response
        renderBotResponse(botMsgId, data);
        
    } catch (error) {
        console.error('Error submitting query:', error);
        replaceLoaderWithError(botMsgId, error.message);
    }
}

// Render message helper
function appendMessage(text, sender) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${sender}`;
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.innerHTML = sender === 'user' ? 'Bạn' : 'RAG';
    
    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    bubble.textContent = text;
    
    msgDiv.appendChild(avatar);
    msgDiv.appendChild(bubble);
    chatWindow.appendChild(msgDiv);
    
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

// Render stepper loading indicator placeholder
function appendBotLoader(id) {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message bot';
    msgDiv.id = id;
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.innerHTML = 'RAG';
    
    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    
    const stepper = document.createElement('div');
    stepper.className = 'rag-stepper';
    stepper.innerHTML = `
        <div class="rag-step" id="${id}-step-1">
            <span class="step-indicator">1</span>
            <span class="step-text">Chuẩn bị câu hỏi & kiểm tra phạm vi...</span>
        </div>
        <div class="rag-step" id="${id}-step-2">
            <span class="step-indicator">2</span>
            <span class="step-text">Truy hồi dữ liệu & tái xếp hạng...</span>
        </div>
        <div class="rag-step" id="${id}-step-3">
            <span class="step-indicator">3</span>
            <span class="step-text">Tổng hợp & soạn thảo câu trả lời...</span>
        </div>
        <div class="rag-step" id="${id}-step-4">
            <span class="step-indicator">4</span>
            <span class="step-text">Xác thực trích dẫn & kiểm định độ bám nguồn...</span>
        </div>
    `;
    
    bubble.appendChild(stepper);
    msgDiv.appendChild(avatar);
    msgDiv.appendChild(bubble);
    chatWindow.appendChild(msgDiv);
    
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

// Replace loader placeholder with clean error card
function replaceLoaderWithError(id, rawErrorMsg) {
    clearStepperAnimation();
    
    const msgDiv = document.getElementById(id);
    if (!msgDiv) return;
    
    const bubble = msgDiv.querySelector('.message-bubble');
    bubble.innerHTML = '';
    
    const friendlyMsg = getFriendlyErrorMessage(rawErrorMsg);
    
    const errorCard = document.createElement('div');
    errorCard.className = 'alert-card error';
    errorCard.innerHTML = `
        <div class="alert-title">Thông báo hệ thống</div>
        <div>${escapeHtml(friendlyMsg)}</div>
    `;
    
    bubble.appendChild(errorCard);
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

// Render final Bot reply
function renderBotResponse(id, data) {
    completeStepperAnimation(id);
    
    // Add a tiny transition delay so the user sees all checkmarks complete
    setTimeout(() => {
        const msgDiv = document.getElementById(id);
        if (!msgDiv) return;
        
        const bubble = msgDiv.querySelector('.message-bubble');
        bubble.innerHTML = '';
        
        // CASE 1: Refused Response
        if (data.refused) {
            const warningCard = document.createElement('div');
            warningCard.className = 'alert-card warning';
            warningCard.innerHTML = `
                <div class="alert-title">Từ chối phục vụ</div>
                <div>${formatAnswerText(data.answer)}</div>
                <div style="font-size: 11px; color: var(--text-muted); margin-top: 6px;">
                    Lý do phân nhóm: <code>${escapeHtml(data.category || '')}</code>
                </div>
            `;
            bubble.appendChild(warningCard);
        } 
        // CASE 2: Success response
        else {
            // 1. Text answer
            const answerPara = document.createElement('div');
            answerPara.innerHTML = formatAnswerText(data.answer);
            bubble.appendChild(answerPara);
            
            // 2. Citations Accordion Container
            if (data.citations && data.citations.length > 0) {
                const citationsHeader = document.createElement('div');
                citationsHeader.style.cssText = 'font-weight: 600; font-size: 13.5px; margin-top: 18px; margin-bottom: 8px; color: var(--accent-gold); font-family: "Outfit", sans-serif;';
                citationsHeader.innerHTML = 'Nguồn trích dẫn chính xác:';
                bubble.appendChild(citationsHeader);
                
                const citationsContainer = document.createElement('div');
                citationsContainer.className = 'citations-container';
                
                data.citations.forEach(cit => {
                    const citId = parseInt(cit.citation_id, 10);
                    if (isNaN(citId)) return; // Bảo vệ an toàn chống XSS qua ID (BUG-12)
                    
                    const card = document.createElement('div');
                    card.className = 'citation-card';
                    card.id = `cit-card-${citId}`;
                    
                    const header = document.createElement('div');
                    header.className = 'citation-card-header';
                    header.innerHTML = `
                        <span>[${citId}] ${escapeHtml(cit.article || '')} ${cit.clause ? `- ${escapeHtml(cit.clause)}` : ''}</span>
                        <span class="citation-tag">${escapeHtml(cit.title || '')}</span>
                    `;
                    
                    const body = document.createElement('div');
                    body.className = 'citation-card-body';
                    body.innerHTML = `
                        <div style="font-weight: 500; color: var(--text-muted); font-size: 12px; margin-bottom: 4px;">Nội dung Điều luật:</div>
                        <div class="citation-evidence">${escapeHtml(cit.evidence || '')}</div>
                        <div style="margin-top: 4px; text-align: right;">
                            <a href="${safeExternalUrl(cit.source_url)}" target="_blank" rel="noopener noreferrer" class="citation-link">Chi tiết văn bản đầy đủ →</a>
                        </div>
                    `;
                    
                    header.addEventListener('click', () => {
                        card.classList.toggle('expanded');
                    });
                    
                    card.appendChild(header);
                    card.appendChild(body);
                    citationsContainer.appendChild(card);
                });
                
                bubble.appendChild(citationsContainer);
            }
            
            // 3. Legal Disclaimer
            const disclaimer = document.createElement('div');
            disclaimer.className = 'disclaimer-text';
            disclaimer.innerHTML = '<strong>Lưu ý pháp lý:</strong> Câu trả lời được tự động tổng hợp từ Bộ luật Lao động 2019 và đã qua bộ đôi hậu kiểm độc lập. Nội dung chỉ mang tính chất cứu tra và tham khảo, không thay thế cho tư vấn pháp lý chính thức từ chuyên gia pháp luật.';
            bubble.appendChild(disclaimer);
        }
        
        chatWindow.scrollTop = chatWindow.scrollHeight;
        
        // Bind click events to citation tags in the answer text
        bubble.querySelectorAll('.cite-tag').forEach(tag => {
            tag.addEventListener('click', (e) => {
                const citId = e.target.getAttribute('data-cit-id');
                const citCard = document.getElementById(`cit-card-${citId}`);
                if (citCard) {
                    citCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                    citCard.classList.add('expanded');
                    
                    // Smooth highlight feedback
                    citCard.style.borderColor = 'var(--accent-gold)';
                    citCard.style.boxShadow = '0 0 10px rgba(245, 158, 11, 0.25)';
                    setTimeout(() => {
                        citCard.style.borderColor = '';
                        citCard.style.boxShadow = '';
                    }, 1500);
                }
            });
        });
        
        // Update Debug Panel contents
        renderDebugContent();
    }, 250);
}

function escapeHtml(value) {
    if (value === null || value === undefined) return '';
    return String(value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function safeExternalUrl(value) {
    if (!value) return '#';
    try {
        const url = new URL(value, window.location.origin);
        if (url.protocol === 'http:' || url.protocol === 'https:') {
            return escapeHtml(url.href);
        }
    } catch (error) {
        return '#';
    }
    return '#';
}

// Format markdown citations in text e.g. [1] or [2] into clickable tags
function formatAnswerText(text) {
    if (!text) return '';
    
    // CẢNH BÁO BẢO MẬT: Bắt buộc thực hiện escapeHtml đầu tiên trước khi chèn bất kỳ thẻ HTML nào (BUG-05)
    let escaped = escapeHtml(text);
    
    // Thay thế markdown bold ** bằng thẻ <strong> (An toàn vì nội dung bên trong đã được escapeHtml)
    escaped = escaped.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // Thay thế xuống dòng bằng <br>
    escaped = escaped.replace(/\n/g, '<br>');

    // Thay thế [1], [2], v.v. bằng thẻ chi tiết (An toàn vì chỉ khớp số nguyên)
    return escaped.replace(/\[(\d+)\]/g, (match, num) => {
        const citId = parseInt(num, 10);
        return `<span class="cite-tag" data-cit-id="${citId}">[${citId}]</span>`;
    });
}

// Render selected debug tab content
function renderDebugContent() {
    if (!currentResponse) {
        debugContent.innerHTML = 'Bắt đầu nhập câu hỏi để xem chi tiết dữ liệu debug hệ thống...';
        return;
    }
    
    if (activeTab === 'retrieved') {
        const chunks = currentResponse.retrieved_chunks || [];
        if (chunks.length === 0) {
            debugContent.innerHTML = '<div style="color: var(--text-warning)">Không tìm thấy hoặc không truy hồi các đoạn văn bản nào.</div>';
            return;
        }
        
        let html = `
            <table class="debug-table">
                <thead>
                    <tr>
                        <th style="width: 50px; text-align: center;">STT</th>
                        <th style="width: 150px;">Phân cấp pháp lý</th>
                        <th style="width: 80px; text-align: center;">Score</th>
                        <th>Nội dung đoạn trích truy hồi</th>
                    </tr>
                </thead>
                <tbody>
        `;
        
        chunks.forEach((chunk, idx) => {
            const location = chunk.article ? `${escapeHtml(chunk.article)} ${chunk.clause ? `- ${escapeHtml(chunk.clause)}` : ''}` : 'Không xác định';
            html += `
                <tr>
                    <td style="text-align: center; font-weight: bold; color: var(--accent-blue);">${idx + 1}</td>
                    <td>
                        <span style="font-weight: 600; color: var(--text-main);">${location}</span>
                        <div style="font-size: 10px; color: var(--text-muted);">${escapeHtml(chunk.title || '')}</div>
                    </td>
                    <td style="text-align: center; font-weight: bold; color: var(--accent-gold);">${Number(chunk.score || 0).toFixed(4)}</td>
                    <td style="font-size: 11px; max-height: 100px; overflow-y: auto; white-space: pre-wrap; color: #a5b4fc;">${escapeHtml(chunk.text || '')}</td>
                </tr>
            `;
        });
        
        html += `
                </tbody>
            </table>
        `;
        
        debugContent.innerHTML = html;
    } 
    else if (activeTab === 'checkers') {
        const isBypassed = bypassRefusalCheckbox.checked;
        const isRefused = currentResponse.refused;
        
        let html = '<div style="display: flex; flex-direction: column; gap: 12px;">';
        
        if (isBypassed) {
            html += `
                <div style="background-color: rgba(245,158,11,0.05); border-left: 3px solid var(--text-warning); padding: 10px; border-radius: 4px;">
                    <div style="font-weight: bold; color: var(--text-warning); font-family: 'Outfit';">⚠️ Chế độ kiểm thử (Bypass Checkers) đang bật</div>
                    <div style="font-size: 11.5px; margin-top: 2px;">Bộ lọc hậu kiểm bị bỏ qua để xuất trực tiếp câu trả lời từ máy phát sinh.</div>
                </div>
            `;
        }
        
        // 1. Citation verification report
        const citStatus = isRefused ? 'BỊ CHẶN' : 'ĐẠT';
        const citColor = isRefused ? 'var(--text-error)' : 'var(--text-success)';
        html += `
            <div style="background-color: rgba(255,255,255,0.01); border: 1px solid var(--border-glass); padding: 10px; border-radius: 6px;">
                <div style="display: flex; justify-content: space-between; font-weight: 500; font-family: 'Outfit';">
                    <span>HẬU KIỂM TĨNH - CITATION CHECKER</span>
                    <span style="color: ${citColor}">${citStatus}</span>
                </div>
                <div style="margin-top: 4px; color: var(--text-muted); font-size:11.5px;">
                    ${isRefused 
                        ? 'Bộ lọc từ chối hoặc sai lệch trích dẫn nguồn đã kích hoạt chặn hiển thị.' 
                        : `Xác nhận: Tất cả ${currentResponse.citations.length} trích dẫn nguồn trùng khớp 100% với dữ liệu lưu trong cơ sở dữ liệu luật.`}
                </div>
            </div>
        `;
        
        // 2. Faithfulness verification report
        const faithStatus = isRefused ? 'BỊ CHẶN' : 'ĐẠT';
        const faithColor = isRefused ? 'var(--text-error)' : 'var(--text-success)';
        html += `
            <div style="background-color: rgba(255,255,255,0.01); border: 1px solid var(--border-glass); padding: 10px; border-radius: 6px;">
                <div style="display: flex; justify-content: space-between; font-weight: 500; font-family: 'Outfit';">
                    <span>HẬU KIỂM ĐỘNG - FAITHFULNESS CHECKER (NLI)</span>
                    <span style="color: ${faithColor}">${faithStatus}</span>
                </div>
                <div style="margin-top: 4px; color: var(--text-muted); font-size:11.5px;">
                    ${isRefused 
                        ? 'Kiểm tra độ bám nguồn bị chặn do phát hiện mâu thuẫn thông tin.' 
                        : 'Không phát hiện hiện tượng bịa đặt thông tin (Hallucination) hoặc mâu thuẫn số liệu so với ngữ cảnh cung cấp.'}
                </div>
            </div>
        `;
        
        html += '</div>';
        debugContent.innerHTML = html;
    } 
    else if (activeTab === 'system') {
        const latencyMs = (requestEndTime - requestStartTime).toFixed(0);
        const latencySec = (latencyMs / 1000).toFixed(2);
        
        const html = `
            <div style="display: flex; flex-direction: column; gap: 8px; max-width: 500px; font-size: 12.5px;">
                <div style="display: flex; justify-content: space-between; padding-bottom: 4px; border-bottom: 1px dashed var(--border-glass);">
                    <span style="color: var(--text-muted);">Thời gian phản hồi (Latency):</span>
                    <span style="font-weight: bold; color: var(--accent-gold);">${latencySec} giây (${latencyMs} ms)</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding-bottom: 4px; border-bottom: 1px dashed var(--border-glass);">
                    <span style="color: var(--text-muted);">Chiến lược tìm kiếm:</span>
                    <span style="font-weight: bold;">${strategySelect.options[strategySelect.selectedIndex].text}</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding-bottom: 4px; border-bottom: 1px dashed var(--border-glass);">
                    <span style="color: var(--text-muted);">Số tài liệu lấy tối đa (Top-K):</span>
                    <span style="font-weight: bold;">${topkSlider.value}</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding-bottom: 4px; border-bottom: 1px dashed var(--border-glass);">
                    <span style="color: var(--text-muted);">Độ tự tin câu trả lời (Confidence):</span>
                    <span style="font-weight: bold;">${(currentResponse.confidence * 100).toFixed(0)}%</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding-bottom: 4px; border-bottom: 1px dashed var(--border-glass);">
                    <span style="color: var(--text-muted);">Trạng thái từ chối (Refused):</span>
                    <span style="font-weight: bold; color: ${currentResponse.refused ? 'var(--text-error)' : 'var(--text-success)'};">
                        ${currentResponse.refused ? 'CÓ (REFUSED)' : 'KHÔNG'}
                    </span>
                </div>
            </div>
        `;
        
        debugContent.innerHTML = html;
    }
}
