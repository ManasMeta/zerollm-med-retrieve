/* app.js */
document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('search-form');
    const input = document.getElementById('search-input');
    const submitBtn = document.querySelector('.search-button');
    const btnText = submitBtn.querySelector('span');
    const loader = submitBtn.querySelector('.loader-spinner');
    
    const resultsContainer = document.getElementById('results-list');
    const metricsBar = document.getElementById('metrics-bar');
    const emptyState = document.getElementById('empty-state');
    
    // Metrics elements
    const mTotal = document.getElementById('metric-total');
    const mBm25 = document.getElementById('metric-bm25');
    const mDense = document.getElementById('metric-dense');
    const mRrf = document.getElementById('metric-rrf');

    // Tab Switching Logic
    const tabSearchBtn = document.getElementById('tab-btn-search');
    const tabAddBtn = document.getElementById('tab-btn-add');
    const searchPanel = document.getElementById('tab-content-search');
    const addPanel = document.getElementById('tab-content-add');

    tabSearchBtn.addEventListener('click', () => {
        tabSearchBtn.classList.add('active');
        tabAddBtn.classList.remove('active');
        searchPanel.classList.remove('hidden');
        addPanel.classList.add('hidden');
        resultsContainer.classList.remove('hidden');
        if (resultsContainer.children.length > 0 || !emptyState.classList.contains('hidden')) {
            // Keep state as is
        }
    });

    tabAddBtn.addEventListener('click', () => {
        tabAddBtn.classList.add('active');
        tabSearchBtn.classList.remove('active');
        addPanel.classList.remove('hidden');
        searchPanel.classList.add('hidden');
        resultsContainer.classList.add('hidden');
        metricsBar.classList.add('hidden');
    });

    // Handle Quick Queries
    document.querySelectorAll('.chip').forEach(chip => {
        chip.addEventListener('click', () => {
            input.value = chip.textContent;
            form.dispatchEvent(new Event('submit'));
        });
    });

    // Handle Document Ingestion Form
    const addForm = document.getElementById('add-document-form');
    const addSubmitBtn = addForm.querySelector('.submit-button');
    const addBtnText = addSubmitBtn.querySelector('span');
    const addLoader = addSubmitBtn.querySelector('.loader-spinner');

    addForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const text = document.getElementById('doc-text').value.trim();
        const question = document.getElementById('doc-question').value.trim();
        const label = document.getElementById('doc-label').value.trim();

        if (!text) return;

        addBtnText.style.display = 'none';
        addLoader.style.display = 'block';
        addSubmitBtn.disabled = true;

        try {
            const response = await fetch('/api/documents', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text, question, label })
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || 'Failed to add document');
            }

            // Show Toast Success Notification
            showToast('Document successfully indexed and persisted!');
            addForm.reset();
            
            // Switch back to search and prefill search with a term from the text
            setTimeout(() => {
                tabSearchBtn.click();
                input.value = text.split(' ').slice(0, 5).join(' ');
            }, 1000);

        } catch (error) {
            console.error('Error adding document:', error);
            alert('Error adding document: ' + error.message);
        } finally {
            addBtnText.style.display = 'block';
            addLoader.style.display = 'none';
            addSubmitBtn.disabled = false;
        }
    });

    function showToast(message) {
        const toast = document.createElement('div');
        toast.className = 'toast-msg';
        toast.textContent = message;
        document.body.appendChild(toast);
        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transition = 'opacity 0.5s ease';
            setTimeout(() => toast.remove(), 500);
        }, 3000);
    }


    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const query = input.value.trim();
        if (!query) return;

        // UI Loading State
        btnText.style.display = 'none';
        loader.style.display = 'block';
        submitBtn.disabled = true;
        
        try {
            const response = await fetch('/api/search', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ query })
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || 'Failed to search');
            }

            const data = await response.json();
            renderResults(data);

        } catch (error) {
            console.error('Search error:', error);
            resultsContainer.innerHTML = `
                <div class="empty-state" style="border-color: rgba(239, 68, 68, 0.3);">
                    <div class="empty-icon" style="color: #ef4444;">⚠️</div>
                    <h3>Error</h3>
                    <p>${error.message}</p>
                </div>
            `;
            emptyState.classList.add('hidden');
            metricsBar.classList.add('hidden');
        } finally {
            // Restore UI
            btnText.style.display = 'block';
            loader.style.display = 'none';
            submitBtn.disabled = false;
        }
    });

    function renderResults(data) {
        // Hide empty state
        emptyState.classList.add('hidden');
        
        // Show and update metrics
        metricsBar.classList.remove('hidden');
        
        mTotal.textContent = `${data.timing.total_ms} ms`;
        mTotal.className = 'metric-value ' + (data.timing.hit_target ? 'success' : 'warning');
        
        mBm25.textContent = `${data.timing.bm25_ms} ms`;
        mDense.textContent = `${data.timing.dense_ms} ms`;
        mRrf.textContent = `${data.timing.rrf_ms} ms`;

        // Render Cards
        resultsContainer.innerHTML = '';
        
        if (data.results.length === 0) {
            resultsContainer.innerHTML = `
                <div class="empty-state">
                    <div class="empty-icon">🤷</div>
                    <h3>No results found</h3>
                    <p>Try rephrasing your query or using more general medical terms.</p>
                </div>
            `;
            return;
        }

        data.results.forEach((item, index) => {
            const delay = index * 0.05; // Staggered animation
            
            const card = document.createElement('div');
            card.className = 'result-card';
            card.style.animationDelay = `${delay}s`;
            
            card.innerHTML = `
                <div class="card-header">
                    <span class="rank-badge">Rank #${index + 1}</span>
                    <span class="score">RRF Score: ${item.rrf_score.toFixed(4)}</span>
                </div>
                <div class="card-text">
                    ${highlightMedicalTerms(item.text)}
                </div>
            `;
            
            resultsContainer.appendChild(card);
        });
    }

    // Simple heuristic to make it look a bit more polished
    function highlightMedicalTerms(text) {
        // Just escapes HTML for safety
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
});
