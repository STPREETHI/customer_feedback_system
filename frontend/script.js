// ENHANCED FRONTEND SCRIPT - WITH ALL FEATURES CORRECTLY MERGED
document.addEventListener('DOMContentLoaded', () => {
    // --- Configuration ---
    const API_BASE_URL = 'http://127.0.0.1:5000';

    // --- Element Selections ---
    const productQueryInput = document.getElementById('product-query-input');
    const analyzeBtn = document.getElementById('analyze-btn');
    const resultsContainer = document.getElementById('results-container');
    const compareQuery1Input = document.getElementById('compare-query1-input');
    const compareQuery2Input = document.getElementById('compare-query2-input');
    const compareBtn = document.getElementById('compare-btn');
    const comparisonResultsContainer = document.getElementById('comparison-results-container');
    const themeSwitcherBtn = document.getElementById('theme-switcher-btn');

    let sentimentChartInstance = null;

    // --- Theme Management ---
    const applyTheme = (theme) => {
        document.body.dataset.theme = theme;
        const sunIcon = themeSwitcherBtn?.querySelector('[data-lucide="sun"]');
        const moonIcon = themeSwitcherBtn?.querySelector('[data-lucide="moon"]');
        if (sunIcon && moonIcon) {
            sunIcon.classList.toggle('hidden', theme === 'dark');
            moonIcon.classList.toggle('hidden', theme === 'light');
        }
    };
    const toggleTheme = () => {
        const currentTheme = document.body.dataset.theme || 'light';
        const newTheme = currentTheme === 'light' ? 'dark' : 'light';
        localStorage.setItem('theme', newTheme);
        applyTheme(newTheme);
    };
    if (themeSwitcherBtn) themeSwitcherBtn.addEventListener('click', toggleTheme);
    applyTheme(localStorage.getItem('theme') || 'light');

    // --- API Fetch Function ---
    async function handleFetch(url, body) {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 90000); // 90s for complex tasks
        try {
            const response = await fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
                body: JSON.stringify(body),
                signal: controller.signal
            });
            clearTimeout(timeoutId);
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || `Server error: ${response.status}`);
            return data;
        } catch (error) {
            clearTimeout(timeoutId);
            if (error.name === 'AbortError') throw new Error('Request timed out. Please try again.');
            throw error;
        }
    }

    // --- UI Rendering (YOUR FULL VERSION) ---
    const ui = {
        setButtonLoading: (button, isLoading, originalText) => {
            if (!button) return;
            button.disabled = isLoading;
            button.innerHTML = isLoading ? `<div class="spinner-small"></div>` : originalText;
        },
        showLoading: (container, message) => {
            if (!container) return;
            container.innerHTML = `<div class="loading-indicator"><div class="spinner"></div><p>${message}</p></div>`;
            container.classList.remove('hidden');
        },
        showError: (container, message) => {
            if (!container) return;
            container.innerHTML = `<div class="loading-indicator"><i data-lucide="alert-triangle" style="color: var(--accent-red);"></i><p>${message}</p></div>`;
            container.classList.remove('hidden');
            lucide.createIcons();
        },
        renderFullAnalysis: (container, data) => {
            const verdictClass = data.verdict.toLowerCase().replace(/\s+/g, '-').replace(' ', '-');
            const advantagesHtml = data.advantages.map(adv => `<li><i data-lucide="check-circle-2"></i> ${adv}</li>`).join('') || '<li>No specific advantages identified.</li>';
            const disadvantagesHtml = data.disadvantages.map(dis => `<li><i data-lucide="x-circle"></i> ${dis}</li>`).join('') || '<li>No specific disadvantages identified.</li>';
            const sourcesHtml = (data.sources || []).map(src => {
                try {
                    return `<li><a href="${src}" target="_blank" rel="noopener noreferrer">${new URL(src).hostname}</a></li>`;
                } catch (e) { return `<li>${src}</li>`; }
            }).join('') || '<li>No sources available.</li>';
            
            container.innerHTML = `
                <div class="dashboard-grid">
                    <div class="grid-item main-summary">
                        <h4>Analysis for ${data.product_name}</h4>
                        <div class="summary-box">
                            <p>${data.summary}</p>
                            <span class="verdict ${verdictClass}">${data.verdict}</span>
                        </div>
                    </div>
                    <div class="grid-item advantages">
                        <h4><i data-lucide="thumbs-up"></i> Advantages</h4>
                        <ul>${advantagesHtml}</ul>
                    </div>
                    <div class="grid-item disadvantages">
                        <h4><i data-lucide="thumbs-down"></i> Disadvantages</h4>
                        <ul>${disadvantagesHtml}</ul>
                    </div>
                    <div class="grid-item word-cloud">
                        <h4><i data-lucide="cloud"></i> Key Topics</h4>
                        <canvas id="wordcloud-canvas"></canvas>
                    </div>
                    <div class="grid-item sentiment-chart">
                        <h4><i data-lucide="pie-chart"></i> Sentiment</h4>
                        <canvas id="sentimentChart"></canvas>
                    </div>
                    <div class="grid-item sources">
                        <h4><i data-lucide="link"></i> Sources</h4>
                        <ul>${sourcesHtml}</ul>
                    </div>
                </div>
                <div class="text-center" style="margin-top: 24px;">
                    <a href="nlp_concepts.html" class="cta-button">
                        <i data-lucide="binary"></i> View NLP Breakdown
                    </a>
                </div>`;
            
            ui.renderSentimentAwareWordCloud('wordcloud-canvas', data.key_topics);
            ui.renderSentimentChart(data.sentiment);
            lucide.createIcons();
        },
        renderSentimentAwareWordCloud: (canvasId, topics) => {
            const canvas = document.getElementById(canvasId);
            if (!canvas || !topics || Object.keys(topics).length === 0) {
                if(canvas) {
                    const ctx = canvas.getContext('2d');
                    ctx.fillStyle = getComputedStyle(document.body).getPropertyValue('--text-secondary');
                    ctx.font = '14px Inter';
                    ctx.textAlign = 'center';
                    ctx.fillText('No topics found', canvas.width/2, canvas.height/2);
                }
                return;
            }
            const list = Object.entries(topics).map(([key, value]) => [key, Math.max(value.count * 10, 12)]);
            if (typeof WordCloud !== 'undefined' && list.length > 0) {
                WordCloud(canvas, {
                    list: list, gridSize: 8, weightFactor: 1.8, fontFamily: 'Inter, sans-serif',
                    color: (word, weight) => {
                        const topicData = topics[word];
                        if (!topicData) return '#6b7280';
                        if (topicData.sentiment > 0.65) return '#10b981';
                        if (topicData.sentiment < 0.35) return '#ef4444';
                        return '#9ca3af';
                    },
                    backgroundColor: 'transparent', rotateRatio: 0, minSize: 10,
                });
            }
        },
        renderSentimentChart: (sentimentData) => {
            const ctx = document.getElementById('sentimentChart')?.getContext('2d');
            if (!ctx) return;
            if(sentimentChartInstance) sentimentChartInstance.destroy();
            const theme = document.body.dataset.theme;
            const chartTextColor = theme === 'dark' ? '#e5e7eb' : '#6b7280';
            const borderColor = theme === 'dark' ? '#1f2937' : '#ffffff';
            sentimentChartInstance = new Chart(ctx, {
                type: 'doughnut', 
                data: {
                    labels: [`Positive (${sentimentData.positive})`, `Negative (${sentimentData.negative})`],
                    datasets: [{ 
                        data: [sentimentData.positive, sentimentData.negative],
                        backgroundColor: ['#10b981', '#ef4444'],
                        borderColor: borderColor, borderWidth: 3, 
                    }]
                },
                options: { 
                    responsive: true, maintainAspectRatio: false, cutout: '70%',
                    plugins: { 
                        legend: { 
                            labels: { color: chartTextColor, font: { size: 12 }, usePointStyle: true },
                            position: 'bottom'
                        } 
                    }
                }
            });
        },
        renderComparisonResult: (container, data) => {
            const renderProductCard = (p) => `
                <div class="comparison-card">
                    <h4>${p.product_name}</h4>
                    <span class="verdict ${p.verdict.toLowerCase().replace(/\s+/g, '-')}">${p.verdict}</span>
                    <h5>Advantages</h5>
                    <ul>${p.advantages.map(adv => `<li><i data-lucide="thumbs-up"></i> ${adv}</li>`).join('')}</ul>
                    <h5>Disadvantages</h5>
                    <ul>${p.disadvantages.map(dis => `<li><i data-lucide="thumbs-down"></i> ${dis}</li>`).join('')}</ul>
                </div>`;
            container.innerHTML = `
                <div class="grid-item main-summary">
                    <h4>Comparison Summary</h4>
                    <div class="summary-box">
                        <p>${data.comparison_summary}</p>
                        <p style="margin-top: 0.75rem; font-weight: 500; color: var(--accent-blue);">
                            <strong>Recommendation:</strong> ${data.recommendation}
                        </p>
                    </div>
                </div>
                <div class="comparison-grid">
                    ${renderProductCard(data.product1)}
                    ${renderProductCard(data.product2)}
                </div>
                <div class="text-center" style="margin-top: 24px;">
                    <a href="compare_nlp.html" class="cta-button">
                        <i data-lucide="binary"></i> View NLP Breakdown
                    </a>
                </div>`;
            lucide.createIcons();
        }
    };

    // --- Suggestion Bot ---
    let lastAnalyzedProduct = null;
    let lastScrapedData = null;
    const suggestionBot = {
        init: () => {
            const botHtml = `<div class="suggestion-bot"><button class="suggestion-trigger" id="suggestion-trigger" title="Get Best in Category"><i data-lucide="lightbulb"></i></button><div class="suggestion-popup" id="suggestion-popup"><h3><i data-lucide="star"></i> Best in Category</h3><p style="font-size: 0.75rem; color: var(--text-secondary); margin: 0 0 0.75rem 0;">Shows the top-rated product in the same category based on review analysis.</p><div class="suggestion-content" id="suggestion-content"><p style="color: var(--text-secondary); font-size: 0.8rem; text-align: center;">Analyze a product first to get category recommendations.</p></div></div></div>`;
            document.body.insertAdjacentHTML('beforeend', botHtml);
            suggestionBot.bindEvents();
        },
        bindEvents: () => {
            const trigger = document.getElementById('suggestion-trigger');
            const popup = document.getElementById('suggestion-popup');
            trigger?.addEventListener('click', () => {
                popup.classList.toggle('active');
                if (lastAnalyzedProduct && lastScrapedData) {
                    suggestionBot.loadSuggestionFromAnalysis();
                }
            });
            document.addEventListener('click', (e) => {
                if (!e.target.closest('.suggestion-bot')) {
                    popup.classList.remove('active');
                }
            });
        },
        loadSuggestionFromAnalysis: async () => {
            const content = document.getElementById('suggestion-content');
            if (!content || !lastAnalyzedProduct) {
                content.innerHTML = '<p>Analyze a product first.</p>'; return;
            }
            content.innerHTML = '<div class="spinner" style="margin: 1rem auto; width: 20px; height: 20px;"></div>';
            try {
                const response = await handleFetch(`${API_BASE_URL}/api/suggest_best_product`, { category: 'general', query: lastAnalyzedProduct });
                if (response.status === 'success' && response.suggestion) {
                    const s = response.suggestion;
                    if (s.rating === 'N/A' || s.name.includes('No') || s.name.includes('research')) {
                        content.innerHTML = `<div class="product-name" style="color: var(--text-secondary);">${s.name}</div><div class="product-reason" style="font-size: 0.8rem; margin-top: 0.5rem;">${s.reason}</div>`;
                    } else {
                        content.innerHTML = `<div class="product-name" style="color: var(--accent-blue); font-weight: 600;">${s.name}</div><div class="product-reason" style="font-size: 0.8rem; margin: 0.5rem 0; line-height: 1.4;">${s.reason}</div><div class="product-rating" style="color: var(--accent-green); font-weight: 500;">Rating: ${s.rating}</div><button onclick="document.getElementById('product-query-input').value='${s.name}'; document.getElementById('analyze-btn').click(); document.getElementById('suggestion-popup').classList.remove('active');" style="margin-top: 0.75rem; padding: 0.4rem 0.75rem; background: var(--accent-green); color: white; border: none; border-radius: 6px; font-size: 0.75rem; cursor: pointer;">📊 Analyze This Product</button>`;
                    }
                } else { throw new Error('Invalid response'); }
            } catch (error) {
                content.innerHTML = `<div style="text-align: center;"><p style="color: var(--accent-red); font-size: 0.8rem;">Unable to load suggestions</p></div>`;
            }
        },
        updateWithNewAnalysis: (productName, scrapedData) => {
            lastAnalyzedProduct = productName;
            lastScrapedData = scrapedData;
        }
    };

    // --- Event Handlers ---
    const handleAnalysis = async () => {
        const query = productQueryInput.value.trim();
        const originalText = analyzeBtn.textContent || 'Analyze';
        if (!query) return;
        
        ui.setButtonLoading(analyzeBtn, true, originalText);
        ui.showLoading(resultsContainer, `Analyzing "${query}"...`);
        
        try {
            const results = await handleFetch(`${API_BASE_URL}/api/analyze_product`, { query });
            ui.renderFullAnalysis(resultsContainer, results);
            localStorage.setItem('latestAnalysis', JSON.stringify(results));
            suggestionBot.updateWithNewAnalysis(query, results);
        } catch (error) { 
            ui.showError(resultsContainer, error.message);
        } finally { 
            ui.setButtonLoading(analyzeBtn, false, originalText);
        }
    };
    
    const handleComparison = async () => {
        const query1 = compareQuery1Input.value.trim();
        const query2 = compareQuery2Input.value.trim();
        const originalText = compareBtn.textContent || 'Generate Comparison';
        if (!query1 || !query2) { ui.showError(comparisonResultsContainer, 'Please provide two product names.'); return; }

        ui.setButtonLoading(compareBtn, true, originalText);
        ui.showLoading(comparisonResultsContainer, `Comparing "${query1}" vs "${query2}"...`);
        
        try {
            const result = await handleFetch(`${API_BASE_URL}/api/compare_products`, { query1, query2 });
            localStorage.setItem('latestComparison', JSON.stringify(result));
            ui.renderComparisonResult(comparisonResultsContainer, result);
        } catch (error) { 
            ui.showError(comparisonResultsContainer, error.message);
        } finally { 
            ui.setButtonLoading(compareBtn, false, originalText);
        }
    };

    // --- NLP Page Logic ---
    function renderNlpColumn(container, data, id_prefix) {
        if (!container || !data) {
            container.innerHTML = `<div class="error">No analysis data found. Please run an analysis first.</div>`;
            return;
        }
        
        const productName = data.product_name || 'The Product';
        const reviewsHtml = (data.classified_reviews || []).map(r => 
            `<li><span class="sentiment-indicator ${r.sentiment}"></span><p>${r.text}</p><span class="sentiment-label ${r.sentiment}">${r.sentiment}</span></li>`
        ).join('') || '<li>No reviews found.</li>';

        // This new version REMOVES the word cloud card from the HTML.
        container.innerHTML = `
            <div class="panel-header"><h3>${productName}</h3></div>

            <div class="concept-card">
                <h4><i data-lucide="message-square"></i> Scraped Reviews & Sentiment</h4>
                <ul class="reviews-list">${reviewsHtml}</ul>
            </div>
            <div class="concept-card">
                <h4><i data-lucide="binary"></i> Sentence Structure Analysis</h4>
                <p>Select a review sentence to visualize its grammatical structure and see its detected sentiment.</p>
                <select id="review-selector-${id_prefix}" class="cfg-selector"></select>
                <button id="visualize-btn-${id_prefix}" class="visualize-btn">Visualize Tree</button>
                <div id="cfg-tree-container-${id_prefix}" class="cfg-tree-container"><p>Select a review and click visualize.</p></div>
            </div>`;
        
        lucide.createIcons();
        
        const reviewSelector = document.getElementById(`review-selector-${id_prefix}`);
        if (data.reviews && data.reviews.length > 0) {
            data.reviews.forEach((review) => {
                const sentence = review.split(/[.!?]/)[0];
                if (sentence && sentence.length > 10) {
                    const option = document.createElement('option');
                    option.value = sentence;
                    option.textContent = `${sentence.substring(0, 70)}...`;
                    reviewSelector.appendChild(option);
                }
            });
        }
        
        document.getElementById(`visualize-btn-${id_prefix}`).addEventListener('click', () => generateCfgTree(id_prefix, data));
        
        // The line that rendered the word cloud has been removed.
    }

    async function generateCfgTree(id_prefix, data) {
        const sentence = document.getElementById(`review-selector-${id_prefix}`).value;
        const container = document.getElementById(`cfg-tree-container-${id_prefix}`);
        if (!sentence) return;

        const reviewData = data.classified_reviews.find(r => r.text.startsWith(sentence));
        const sentiment = reviewData ? reviewData.sentiment : 'Unknown';
        const sentimentHtml = `<p style="text-align:center; margin-bottom: 1rem; font-size: 0.9rem;">Detected Sentiment: <span class="sentiment-label ${sentiment}">${sentiment}</span></p>`;

        container.innerHTML = `<div class="loader"></div>`;
        try {
            const responseData = await handleFetch(`${API_BASE_URL}/api/parse_sentence`, { sentence });
            container.innerHTML = sentimentHtml + `<div class="tree">${createTreeHtml(responseData.parsed_tree)}</div>`;
        } catch (error) {
            container.innerHTML = `<div class="error">${error.message}</div>`;
        }
    }

    function createTreeHtml(node) {
        if (!node) return '';
        let childrenHtml = (node.children && node.children.length > 0) ? `<ul>${node.children.map(child => `<li>${createTreeHtml(child)}</li>`).join('')}</ul>` : '';
        return `<span class="node">${node.word} <small>(${node.label})</small></span>${childrenHtml}`;
    }

    // --- Page Initialization ---
    function initializePage() {
        if (analyzeBtn) analyzeBtn.addEventListener('click', handleAnalysis);
        if (productQueryInput) productQueryInput.addEventListener('keypress', (e) => { if (e.key === 'Enter') handleAnalysis(); });
        if (compareBtn) compareBtn.addEventListener('click', handleComparison);

        if (document.getElementById('nlp-main-content')) {
            const analysisData = JSON.parse(localStorage.getItem('latestAnalysis'));
            renderNlpColumn(document.getElementById('nlp-main-content'), analysisData, 'single');
        }
        if (document.getElementById('compare-nlp-main-content')) {
            const comparisonData = JSON.parse(localStorage.getItem('latestComparison'));
            const container1 = document.getElementById('product1-nlp-column');
            const container2 = document.getElementById('product2-nlp-column');
            if (!comparisonData || !container1 || !container2) {
                document.getElementById('nlp-subtitle').textContent = 'Please run a comparison on the Compare page first.';
            } else {
                renderNlpColumn(container1, comparisonData.product1, 'p1');
                renderNlpColumn(container2, comparisonData.product2, 'p2');
            }
        }
        suggestionBot.init();
        lucide.createIcons();
    }

    // --- Run Initialization ---
    initializePage();
});