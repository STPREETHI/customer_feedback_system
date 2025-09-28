// ENHANCED FRONTEND SCRIPT - FINAL DEFINITIVE VERSION
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
        const timeoutId = setTimeout(() => controller.abort(), 45000); // 45s timeout
        
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
                throw new Error('Analysis timed out. The server may be busy or the request is too complex.');
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
                        <h4>The Analyst's Verdict for ${data.product_name}</h4>
                        <p>${data.summary}</p>
                        <span class="verdict ${verdictClass}">${data.verdict}</span>
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
                         <h4>Key Topic Cloud</h4>
                         <canvas id="wordcloud-canvas" width="600" height="250"></canvas>
                    </div>
                    <div class="grid-item sentiment-chart">
                         <h4>Sentiment</h4>
                          <canvas id="sentimentChart"></canvas>
                    </div>
                     <div class="grid-item sources">
                         <h4><i data-lucide="link"></i> Sources Analyzed</h4>
                        <ul>${sourcesHtml}</ul>
                    </div>
                </div>`;
            
            ui.renderSentimentAwareWordCloud(data.key_topics);
            ui.renderSentimentChart(data.sentiment);
            lucide.createIcons();
        },
        renderSentimentAwareWordCloud: (topics) => {
            const canvas = document.getElementById('wordcloud-canvas');
            if (!canvas || !topics || Object.keys(topics).length === 0) return;
            const list = Object.entries(topics).map(([key, value]) => [key, value.count * 10]);
            WordCloud(canvas, {
                list: list, gridSize: Math.round(16 * canvas.width / 1024),
                weightFactor: 4, fontFamily: 'Inter, sans-serif',
                color: (word, weight) => {
                    const topicData = topics[word];
                    const theme = document.body.dataset.theme;
                    if (!topicData) return theme === 'dark' ? '#e5e7eb' : '#1f2937';
                    if (topicData.sentiment > 0.65) return '#10b981';
                    if (topicData.sentiment < 0.35) return '#ef4444';
                    return theme === 'dark' ? '#9ca3af' : '#6b7280';
                },
                backgroundColor: 'transparent', rotateRatio: 0, minSize: 10
            });
        },
        renderSentimentChart: (sentimentData) => {
            const ctx = document.getElementById('sentimentChart')?.getContext('2d');
            if (!ctx) return;
            if(sentimentChartInstance) sentimentChartInstance.destroy();
            const theme = document.body.dataset.theme;
            const chartTextColor = theme === 'dark' ? '#e5e7eb' : '#6b7280';
            const borderColor = theme === 'dark' ? '#1f2937' : '#ffffff';
            sentimentChartInstance = new Chart(ctx, {
                type: 'doughnut', data: {
                    labels: [`Positive (${sentimentData.positive})`, `Negative (${sentimentData.negative})`],
                    datasets: [{ data: [sentimentData.positive, sentimentData.negative],
                        backgroundColor: [ '#28a745', '#ef4444'],
                        borderColor: borderColor, 
                        borderWidth: 4, 
                    }]
                },
                options: { responsive: true, maintainAspectRatio: false, cutout: '70%',
                    plugins: { legend: { labels: { color: chartTextColor } } }
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
                <div class="main-summary">
                    <h4>The Final Verdict</h4>
                    <p>${data.comparison_summary.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')}</p>
                </div>
                <div class="comparison-grid">
                    ${renderProductCard(data.product1)}
                    ${renderProductCard(data.product2)}
                </div>`;
            lucide.createIcons();
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
        if (!query1 || !query2) { ui.showError(comparisonResultsContainer, 'Please provide two queries.'); return; }

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
    
    // --- Initializers & Event Listeners ---
    if (analyzeBtn) analyzeBtn.addEventListener('click', handleAnalysis);
    if (productQueryInput) {
        productQueryInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') handleAnalysis();
        });
    }
    if (compareBtn) compareBtn.addEventListener('click', handleComparison);

    lucide.createIcons();
});

