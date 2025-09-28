// ENHANCED FRONTEND SCRIPT - WITH SUGGESTION BOT
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
    const sunIcon = themeSwitcherBtn?.querySelector('[data-lucide="sun"]');
    const moonIcon = themeSwitcherBtn?.querySelector('[data-lucide="moon"]');
    
    let sentimentChartInstance = null;

    // --- Theme Management ---
    const applyTheme = (theme) => {
        document.body.dataset.theme = theme;
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

    if (themeSwitcherBtn) {
        themeSwitcherBtn.addEventListener('click', toggleTheme);
    }
    const savedTheme = localStorage.getItem('theme') || 'light';
    applyTheme(savedTheme);

    // --- API Fetch Function ---
    async function handleFetch(url, body) {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 60000); // 60s timeout for optimized backend
        
        try {
            const response = await fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
                body: JSON.stringify(body),
                signal: controller.signal
            });
            clearTimeout(timeoutId);
            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.error || `Server error: ${response.status}`);
            }
            return data;
        } catch (error) {
            clearTimeout(timeoutId);
            if (error.name === 'AbortError') {
                throw new Error('Analysis timed out. Please try again with a simpler query.');
            }
            throw error;
        }
    }
    
    // --- UI Rendering ---
    const ui = {
        setButtonLoading: (button, isLoading, originalText) => {
            if (!button) return;
            button.disabled = isLoading;
            button.innerHTML = isLoading ? '<div class="spinner-small"></div>' : originalText;
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
            const verdictClass = data.verdict.toLowerCase().replace(/\s+/g, '-');
            const advantagesHtml = data.advantages.map(adv => `<li><i data-lucide="check-circle-2"></i> ${adv}</li>`).join('') || '<li>No specific advantages identified.</li>';
            const disadvantagesHtml = data.disadvantages.map(dis => `<li><i data-lucide="x-circle"></i> ${dis}</li>`).join('') || '<li>No specific disadvantages identified.</li>';
            const sourcesHtml = (data.sources || []).map(src => `<li><a href="${src}" target="_blank" rel="noopener noreferrer">${new URL(src).hostname}</a></li>`).join('') || '<li>No sources available.</li>';
            
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
                         <canvas id="wordcloud-canvas" width="400" height="140"></canvas>
                    </div>
                    <div class="grid-item sentiment-chart">
                         <h4><i data-lucide="pie-chart"></i> Sentiment</h4>
                          <canvas id="sentimentChart"></canvas>
                    </div>
                     <div class="grid-item sources">
                         <h4><i data-lucide="link"></i> Sources</h4>
                        <ul>${sourcesHtml}</ul>
                    </div>
                </div>`;
            
            ui.renderSentimentAwareWordCloud(data.key_topics);
            ui.renderSentimentChart(data.sentiment);
            lucide.createIcons();
        },
        renderSentimentAwareWordCloud: (topics) => {
            const canvas = document.getElementById('wordcloud-canvas');
            if (!canvas || !topics || Object.keys(topics).length === 0) {
                if (canvas) {
                    const ctx = canvas.getContext('2d');
                    ctx.fillStyle = 'var(--text-secondary)';
                    ctx.font = '14px Inter';
                    ctx.textAlign = 'center';
                    ctx.fillText('No topics found', canvas.width/2, canvas.height/2);
                }
                return;
            }
            
            canvas.width = 400;
            canvas.height = 140;
            
            const list = Object.entries(topics).map(([key, value]) => [key, Math.max(value.count * 6, 10)]);
            
            if (typeof WordCloud !== 'undefined' && list.length > 0) {
                WordCloud(canvas, {
                    list: list,
                    gridSize: 6,
                    weightFactor: 1.5,
                    fontFamily: 'Inter, sans-serif',
                    color: (word, weight) => {
                        const topicData = topics[word];
                        if (!topicData) return '#6b7280';
                        if (topicData.sentiment > 0.65) return '#10b981';
                        if (topicData.sentiment < 0.35) return '#ef4444';
                        return '#6b7280';
                    },
                    backgroundColor: 'transparent',
                    rotateRatio: 0,
                    minSize: 10,
                    maxSize: 20,
                    drawOutOfBound: false
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
                        backgroundColor: [ '#10b981', '#ef4444'],
                        borderColor: borderColor, 
                        borderWidth: 2, 
                    }]
                },
                options: { 
                    responsive: true, 
                    maintainAspectRatio: false, 
                    cutout: '60%',
                    plugins: { 
                        legend: { 
                            labels: { 
                                color: chartTextColor,
                                font: { size: 11 },
                                usePointStyle: true,
                            },
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
                </div>
            `;
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
                </div>`;
            lucide.createIcons();
        }
    };

    let lastAnalyzedProduct = null;
    let lastScrapedData = null;

    // --- Suggestion Bot Implementation ---
    const suggestionBot = {
        init: () => {
            const botHtml = `
                <div class="suggestion-bot">
                    <button class="suggestion-trigger" id="suggestion-trigger" title="Get Best in Category">
                        <i data-lucide="lightbulb"></i>
                    </button>
                    <div class="suggestion-popup" id="suggestion-popup">
                        <h3><i data-lucide="star"></i> Best in Category</h3>
                        <p style="font-size: 0.75rem; color: var(--text-secondary); margin: 0 0 0.75rem 0;">
                            Shows the top-rated product in the same category based on review analysis.
                        </p>
                        <div class="suggestion-content" id="suggestion-content">
                            <p style="color: var(--text-secondary); font-size: 0.8rem; text-align: center;">
                                Analyze a product first to get category recommendations.
                            </p>
                        </div>
                    </div>
                </div>
            `;
            document.body.insertAdjacentHTML('beforeend', botHtml);
            
            suggestionBot.bindEvents();
        },
        
        bindEvents: () => {
            const trigger = document.getElementById('suggestion-trigger');
            const popup = document.getElementById('suggestion-popup');
            
            trigger?.addEventListener('click', () => {
                popup.classList.toggle('active');
                // Auto-load suggestion if we have analyzed product
                if (lastAnalyzedProduct && lastScrapedData) {
                    suggestionBot.loadSuggestionFromAnalysis();
                }
            });
            
            // Close popup when clicking outside
            document.addEventListener('click', (e) => {
                if (!e.target.closest('.suggestion-bot')) {
                    popup.classList.remove('active');
                }
            });
        },
        
        loadSuggestionFromAnalysis: async () => {
            const content = document.getElementById('suggestion-content');
            if (!content || !lastAnalyzedProduct) {
                content.innerHTML = '<p style="color: var(--text-secondary); font-size: 0.8rem; text-align: center;">Analyze a product first to get recommendations.</p>';
                return;
            }
            
            content.innerHTML = '<div class="spinner" style="margin: 1rem auto; width: 20px; height: 20px;"></div>';
            
            try {
                const response = await handleFetch(`${API_BASE_URL}/api/suggest_best_product`, { 
                    category: 'general', 
                    query: lastAnalyzedProduct 
                });
                
                if (response.status === 'success' && response.suggestion) {
                    const suggestion = response.suggestion;
                    
                    if (suggestion.rating === 'N/A' || suggestion.name.includes('No') || suggestion.name.includes('research')) {
                        content.innerHTML = `
                            <div class="product-name" style="color: var(--text-secondary);">${suggestion.name}</div>
                            <div class="product-reason" style="font-size: 0.8rem; margin-top: 0.5rem;">${suggestion.reason}</div>
                        `;
                    } else {
                        content.innerHTML = `
                            <div class="product-name" style="color: var(--accent-blue); font-weight: 600;">${suggestion.name}</div>
                            <div class="product-reason" style="font-size: 0.8rem; margin: 0.5rem 0; line-height: 1.4;">${suggestion.reason}</div>
                            <div class="product-rating" style="color: var(--accent-green); font-weight: 500;">Rating: ${suggestion.rating}</div>
                            <button onclick="
                                document.getElementById('product-query-input').value='${suggestion.name}';
                                document.getElementById('analyze-btn').click();
                                document.getElementById('suggestion-popup').classList.remove('active');
                            " 
                            style="
                                margin-top: 0.75rem; 
                                padding: 0.4rem 0.75rem; 
                                background: var(--accent-green); 
                                color: white; 
                                border: none; 
                                border-radius: 6px; 
                                font-size: 0.75rem; 
                                cursor: pointer;
                                transition: background-color 0.2s ease;
                            "
                            onmouseover="this.style.background='#059669'"
                            onmouseout="this.style.background='var(--accent-green)'">
                                📊 Analyze This Product
                            </button>
                        `;
                    }
                } else {
                    throw new Error('Invalid response format');
                }
            } catch (error) {
                console.error('Suggestion error:', error);
                content.innerHTML = `
                    <div style="text-align: center;">
                        <p style="color: var(--accent-red); font-size: 0.8rem;">Unable to load suggestions</p>
                        <p style="color: var(--text-secondary); font-size: 0.75rem;">Try analyzing the product again</p>
                    </div>
                `;
            }
        },
        
        updateWithNewAnalysis: (productName, scrapedData) => {
            lastAnalyzedProduct = productName;
            lastScrapedData = scrapedData;
        }
    };

    // --- Event Handlers ---
    const handleAnalysis = async () => {
        if (!productQueryInput || !resultsContainer || !analyzeBtn) return;
        const query = productQueryInput.value.trim();
        const originalText = analyzeBtn.textContent || 'Analyze';
        if (!query) { ui.showError(resultsContainer, 'Please enter a product name or URL.'); return; }
        
        ui.setButtonLoading(analyzeBtn, true, originalText);
        ui.showLoading(resultsContainer, `Analyzing "${query}"... This may take a moment.`);
        
        try {
            const results = await handleFetch(`${API_BASE_URL}/api/analyze_product`, { query });
            ui.renderFullAnalysis(resultsContainer, results);
            
            // Update suggestion bot with new analysis
            suggestionBot.updateWithNewAnalysis(query, results);
            
        } catch (error) { 
            ui.showError(resultsContainer, error.message);
        } finally { 
            ui.setButtonLoading(analyzeBtn, false, originalText);
        }
    };
    
    const handleComparison = async () => {
        if (!compareQuery1Input || !compareQuery2Input || !comparisonResultsContainer || !compareBtn) return;
        const query1 = compareQuery1Input.value.trim();
        const query2 = compareQuery2Input.value.trim();
        const originalText = compareBtn.textContent || 'Generate Comparison';
        if (!query1 || !query2) { ui.showError(comparisonResultsContainer, 'Please provide two product names.'); return; }

        ui.setButtonLoading(compareBtn, true, originalText);
        ui.showLoading(comparisonResultsContainer, `Comparing "${query1}" vs "${query2}"...`);
        
        try {
            const result = await handleFetch(`${API_BASE_URL}/api/compare_products`, { query1, query2 });
            ui.renderComparisonResult(comparisonResultsContainer, result);
        } catch (error) { 
            ui.showError(comparisonResultsContainer, error.message);
        } finally { 
            ui.setButtonLoading(compareBtn, false, originalText);
        }
    };
    
    // --- Initialize Everything ---
    if (analyzeBtn) analyzeBtn.addEventListener('click', handleAnalysis);
    if (productQueryInput) {
        productQueryInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') handleAnalysis();
        });
    }
    if (compareBtn) compareBtn.addEventListener('click', handleComparison);

    // Initialize suggestion bot
    suggestionBot.init();
    
    lucide.createIcons();
});