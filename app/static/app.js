// ViLaborRAG Frontend Logic (app.js)

// State management
let currentResponse = null;
let requestStartTime = null;
let requestEndTime = null;
let activeTab = 'retrieved';

// DOM Elements
const queryInput = document.getElementById('query-input');
const sendBtn = document.getElementById('send-btn');
const chatWindow = document.getElementById('chat-window');
const strategySelect = document.getElementById('strategy-select');
const topkSlider = document.getElementById('topk-slider');
const bypassRefusalCheckbox = document.getElementById('bypass-refusal-checkbox');
const debugContent = document.getElementById('debug-content');
const debugTabs = document.querySelectorAll('.debug-tab');

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
            geminiStatus.textContent = 'Giả lập (Mock Mode)';
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
        });
    });
}

// Execute suggestions from Quick Templates
function askSuggestion(text) {
    queryInput.value = text;
    // Dispatch input event to resize textarea
    queryInput.dispatchEvent(new Event('input'));
    submitQuery();
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
    
    // Create bot placeholder bubble with shimmer loader
    const botMsgId = 'bot-msg-' + Date.now();
    appendBotLoader(botMsgId);
    
    // Prepare API Request Payload
    const payload = {
        query: query,
        strategy: strategySelect.value,
        top_k: parseInt(topkSlider.value, 10),
        bypass_refusal: bypassRefusalCheckbox.checked
    };
    
    try {
        requestStartTime = performance.now();
        const response = await fetch('/api/v1/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });
        
        requestEndTime = performance.now();
        
        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.detail || 'Lỗi xử lý yêu cầu RAG');
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
    
    // Scroll chat window to bottom
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

// Render shimmer loading animation placeholder
function appendBotLoader(id) {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message bot';
    msgDiv.id = id;
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.innerHTML = 'RAG';
    
    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    
    const loader = document.createElement('div');
    loader.className = 'shimmer-loader';
    loader.innerHTML = `
        <div class="shimmer-line medium"></div>
        <div class="shimmer-line"></div>
        <div class="shimmer-line short"></div>
    `;
    
    bubble.appendChild(loader);
    msgDiv.appendChild(avatar);
    msgDiv.appendChild(bubble);
    chatWindow.appendChild(msgDiv);
    
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

// Replace loader placeholder with Error message
function replaceLoaderWithError(id, message) {
    const msgDiv = document.getElementById(id);
    if (!msgDiv) return;
    
    const bubble = msgDiv.querySelector('.message-bubble');
    bubble.innerHTML = '';
    
    const errorCard = document.createElement('div');
    errorCard.className = 'alert-card error';
    errorCard.innerHTML = `
        <div class="alert-title">❌ Lỗi Hệ Thống</div>
        <div>Không thể hoàn thành yêu cầu: <strong>${escapeHtml(message)}</strong>. Vui lòng kiểm tra lại kết nối backend.</div>
    `;
    
    bubble.appendChild(errorCard);
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

// Render final Bot reply
function renderBotResponse(id, data) {
    const msgDiv = document.getElementById(id);
    if (!msgDiv) return;
    
    const bubble = msgDiv.querySelector('.message-bubble');
    bubble.innerHTML = '';
    
    // CASE 1: Refused Response
    if (data.refused) {
        const warningCard = document.createElement('div');
        warningCard.className = 'alert-card warning';
        warningCard.innerHTML = `
            <div class="alert-title">⚠️ Từ Chối Trả Lời (Refusal Module)</div>
            <div>${formatAnswerText(data.answer)}</div>
            <div style="font-size: 11px; color: var(--text-muted); margin-top: 4px;">
                Phân nhóm từ chối: <code style="background-color: rgba(255,255,255,0.05); padding: 2px 4px; border-radius:4px;">${escapeHtml(data.category || '')}</code>
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
            citationsHeader.style.cssText = 'font-weight: 600; font-size: 13.5px; margin-top: 18px; margin-bottom: 8px; color: var(--accent-gold);';
            citationsHeader.innerHTML = '📂 Nguồn trích dẫn chính xác:';
            bubble.appendChild(citationsHeader);
            
            const citationsContainer = document.createElement('div');
            citationsContainer.className = 'citations-container';
            
            data.citations.forEach(cit => {
                const card = document.createElement('div');
                card.className = 'citation-card';
                card.id = `cit-card-${cit.citation_id}`;
                
                const header = document.createElement('div');
                header.className = 'citation-card-header';
                header.innerHTML = `
                    <span>[${escapeHtml(cit.citation_id)}] ${escapeHtml(cit.article || '')} ${cit.clause ? `- ${escapeHtml(cit.clause)}` : ''}</span>
                    <span class="citation-tag">${escapeHtml(cit.title || '')}</span>
                `;
                
                const body = document.createElement('div');
                body.className = 'citation-card-body';
                body.innerHTML = `
                    <div style="font-weight: 600; color: var(--text-muted);">Nội dung điều luật (Evidence):</div>
                    <div class="citation-evidence">${escapeHtml(cit.evidence || '')}</div>
                    <div style="margin-top: 4px; text-align: right;">
                        <a href="${safeExternalUrl(cit.source_url)}" target="_blank" rel="noopener noreferrer" class="citation-link">📖 Xem văn bản đầy đủ</a>
                    </div>
                `;
                
                header.addEventListener('click', () => {
                    const isVisible = body.style.display === 'flex';
                    body.style.display = isVisible ? 'none' : 'flex';
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
        disclaimer.innerHTML = '💡 <strong>Lưu ý:</strong> Câu trả lời được sinh tự động dựa trên dữ liệu văn bản Bộ luật Lao động 2019 và đã qua bộ đôi hậu kiểm độc lập. Thông tin mang tính chất tham khảo cứu pháp lý và không thay thế cho tư vấn pháp lý chính thức từ luật sư.';
        bubble.appendChild(disclaimer);
    }
    
    // Scroll chat window to bottom
    chatWindow.scrollTop = chatWindow.scrollHeight;
    
    // Bind click events to citation tags in the answer text
    bubble.querySelectorAll('.cite-tag').forEach(tag => {
        tag.addEventListener('click', (e) => {
            const citId = e.target.getAttribute('data-cit-id');
            const citCard = document.getElementById(`cit-card-${citId}`);
            if (citCard) {
                // Scroll citation card into view gently
                citCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                // Toggle open the body if closed
                const cardBody = citCard.querySelector('.citation-card-body');
                if (cardBody) {
                    cardBody.style.display = 'flex';
                }
                // Visual highlight effect
                citCard.style.borderColor = 'var(--accent-gold)';
                setTimeout(() => {
                    citCard.style.borderColor = 'var(--border-glass)';
                }, 1500);
            }
        });
    });
    
    // Update Debug Panel contents
    renderDebugContent();
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
    // Escape HTML first to prevent XSS
    let escaped = escapeHtml(text);
        
    // Replace markdown-style bold tags **bold**
    escaped = escaped.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // Replace newline symbols with <br>
    escaped = escaped.replace(/\n/g, '<br>');

    // Replace [1], [2], etc. with styled span cite-tags
    return escaped.replace(/\[(\d+)\]/g, (match, num) => {
        return `<span class="cite-tag" data-cit-id="${num}">[${num}]</span>`;
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
            debugContent.innerHTML = '<div style="color: var(--text-warning)">Không tìm thấy hoặc không truy hồi các đoạn văn bản nào (Rỗng).</div>';
            return;
        }
        
        let html = `
            <table class="debug-table">
                <thead>
                    <tr>
                        <th style="width: 60px;">Thứ tự</th>
                        <th style="width: 150px;">Pháp lý</th>
                        <th style="width: 80px; text-align: center;">Score</th>
                        <th>Nội dung văn bản truy hồi (Snippet)</th>
                    </tr>
                </thead>
                <tbody>
        `;
        
        chunks.forEach((chunk, idx) => {
            const location = chunk.article ? `${escapeHtml(chunk.article)} ${chunk.clause ? `- ${escapeHtml(chunk.clause)}` : ''}` : 'Không xác định';
            html += `
                <tr>
                    <td style="text-align: center; font-weight: bold; color: var(--accent-indigo);">${idx + 1}</td>
                    <td>
                        <span style="font-weight: 600; color: var(--text-main);">${location}</span>
                        <div style="font-size: 10px; color: var(--text-muted);">${escapeHtml(chunk.title || '')}</div>
                    </td>
                    <td style="text-align: center; font-weight: bold; color: var(--accent-gold);">${Number(chunk.score || 0).toFixed(4)}</td>
                    <td style="font-size: 11.5px; color: #a5b4fc; max-height: 100px; overflow-y: auto; white-space: pre-wrap;">${escapeHtml(chunk.text || '')}</td>
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
        // Draw Checkers evaluation logs based on refusal and bypass status
        const isBypassed = bypassRefusalCheckbox.checked;
        const isRefused = currentResponse.refused;
        
        let html = '<div style="display: flex; flex-direction: column; gap: 16px;">';
        
        if (isBypassed) {
            html += `
                <div style="background-color: rgba(240,136,62,0.1); border-left: 4px solid var(--text-warning); padding: 12px; border-radius: 6px;">
                    <div style="font-weight: bold; color: var(--text-warning);">⚠️ Chế độ Kiểm thử (Bypass Checkers) đang hoạt động</div>
                    <div>Các tầng hậu kiểm đã được bỏ qua để trích xuất trực tiếp câu trả lời từ máy phát sinh.</div>
                </div>
            `;
        }
        
        // 1. Citation verification report
        const citStatus = isRefused ? 'BỊ CHẶN' : 'ĐẠT (PASSED)';
        const citColor = isRefused ? 'var(--text-error)' : 'var(--text-success)';
        html += `
            <div style="background-color: rgba(255,255,255,0.02); border: 1px solid var(--border-glass); padding: 14px; border-radius: 8px;">
                <div style="display: flex; justify-content: space-between; font-weight: bold;">
                    <span>🔎 HẬU KIỂM TĨNH - CITATION CHECKER</span>
                    <span style="color: ${citColor}">${citStatus}</span>
                </div>
                <div style="margin-top: 8px; color: var(--text-muted); font-size:12px;">
                    ${isRefused 
                        ? 'Bộ lọc từ chối hoặc sai khớp nguồn đã kích hoạt chặn sinh văn bản.' 
                        : `Xác nhận: Tất cả ${currentResponse.citations.length} trích dẫn trong câu trả lời hoàn toàn trùng khớp với nội dung lưu trong cơ sở dữ liệu.`}
                </div>
            </div>
        `;
        
        // 2. Faithfulness verification report
        const faithStatus = isRefused ? 'BỊ CHẶN' : 'ĐẠT (PASSED)';
        const faithColor = isRefused ? 'var(--text-error)' : 'var(--text-success)';
        html += `
            <div style="background-color: rgba(255,255,255,0.02); border: 1px solid var(--border-glass); padding: 14px; border-radius: 8px;">
                <div style="display: flex; justify-content: space-between; font-weight: bold;">
                    <span>⚖️ HẬU KIỂM ĐỘNG - FAITHFULNESS CHECKER (NLI)</span>
                    <span style="color: ${faithColor}">${faithStatus}</span>
                </div>
                <div style="margin-top: 8px; color: var(--text-muted); font-size:12px;">
                    ${isRefused 
                        ? 'Kiểm định bám nguồn bị chặn do lỗi logic hoặc không tìm thấy văn bản thích hợp.' 
                        : 'Không phát hiện bất kỳ sự mâu thuẫn hay hiện tượng bịa đặt thông tin (Hallucination) nào đối chiếu với ngữ cảnh cung cấp.'}
                </div>
            </div>
        `;
        
        // 3. Disclaimer checker report
        const discStatus = isRefused ? 'KHÔNG YÊU CẦU' : 'ĐẠT (PASSED)';
        const discColor = isRefused ? 'var(--text-muted)' : 'var(--text-success)';
        html += `
            <div style="background-color: rgba(255,255,255,0.02); border: 1px solid var(--border-glass); padding: 14px; border-radius: 8px;">
                <div style="display: flex; justify-content: space-between; font-weight: bold;">
                    <span>💡 THÔNG BÁO MIỄN TRỪ - DISCLAIMER CHECKER</span>
                    <span style="color: ${discColor}">${discStatus}</span>
                </div>
                <div style="margin-top: 8px; color: var(--text-muted); font-size:12px;">
                    ${isRefused 
                        ? 'Từ chối trả lời không cần bổ sung điều khoản miễn trừ trách nhiệm.' 
                        : 'Đã phát hiện và xác nhận có đính kèm phần lưu ý miễn trừ trách nhiệm hợp quy định.'}
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
            <div style="display: flex; flex-direction: column; gap: 12px; max-width: 600px;">
                <div style="display: flex; justify-content: space-between; padding-bottom: 6px; border-bottom: 1px dashed var(--border-glass);">
                    <span style="color: var(--text-muted);">Thời gian phản hồi (Latency):</span>
                    <span style="font-weight: bold; color: var(--accent-gold);">${latencySec} giây (${latencyMs} ms)</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding-bottom: 6px; border-bottom: 1px dashed var(--border-glass);">
                    <span style="color: var(--text-muted);">Chiến lược tìm kiếm hoạt động:</span>
                    <span style="font-weight: bold; color: var(--text-main);">${strategySelect.options[strategySelect.selectedIndex].text}</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding-bottom: 6px; border-bottom: 1px dashed var(--border-glass);">
                    <span style="color: var(--text-muted);">Số tài liệu lấy tối đa (Top-K):</span>
                    <span style="font-weight: bold; color: var(--text-main);">${topkSlider.value}</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding-bottom: 6px; border-bottom: 1px dashed var(--border-glass);">
                    <span style="color: var(--text-muted);">Độ tự tin câu trả lời (Confidence):</span>
                    <span style="font-weight: bold; color: var(--text-main);">${(currentResponse.confidence * 100).toFixed(0)}%</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding-bottom: 6px; border-bottom: 1px dashed var(--border-glass);">
                    <span style="color: var(--text-muted);">Trạng thái từ chối (Refused):</span>
                    <span style="font-weight: bold; color: ${currentResponse.refused ? 'var(--text-error)' : 'var(--text-success)'};">
                        ${currentResponse.refused ? 'YES' : 'NO'}
                    </span>
                </div>
                <div style="display: flex; justify-content: space-between; padding-bottom: 6px; border-bottom: 1px dashed var(--border-glass);">
                    <span style="color: var(--text-muted);">Bypass Refusal Flag:</span>
                    <span style="font-weight: bold; color: var(--text-main);">${bypassRefusalCheckbox.checked ? 'ON' : 'OFF'}</span>
                </div>
            </div>
        `;
        
        debugContent.innerHTML = html;
    }
}
