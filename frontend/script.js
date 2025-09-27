// This is the definitive and final frontend script with the futuristic UI.
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
    
    let sentimentChartInstance = null;

    // --- API Fetch Function ---
    async function handleFetch(url, body) {
        const response = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || `Server responded with status: ${response.status}`);
        }
        return data;
    }
    
    // --- UI Rendering ---
    const ui = {
        setButtonLoading: (button, isLoading, icon = 'arrow-right') => {
            if (!button) return;
            button.disabled = isLoading;
            button.innerHTML = isLoading ? '<div class="spinner-small"></div>' : (icon.includes('<') ? icon : `<i data-lucide="${icon}"></i>`);
            if(!isLoading) lucide.createIcons();
        },
        showLoading: (container, message) => {
            if (!container) return;
            container.innerHTML = `<div class="loading-indicator glass-effect"><div class="spinner"></div><p>${message}</p></div>`;
            container.classList.remove('hidden');
        },
        showError: (container, message) => {
            if (!container) return;
            container.innerHTML = `<div class="loading-indicator glass-effect"><i data-lucide="alert-triangle" style="color: var(--accent-red);"></i><p>${message}</p></div>`;
            container.classList.remove('hidden');
            lucide.createIcons();
        },
        renderFullAnalysis: (container, data) => {
            const advantagesHtml = data.advantages.map(adv => `<li><i data-lucide="check-circle-2"></i> ${adv}</li>`).join('');
            const disadvantagesHtml = data.disadvantages.map(dis => `<li><i data-lucide="x-circle"></i> ${dis}</li>`).join('');
            const sourcesHtml = (data.sources || []).map(src => `<li><a href="${src}" target="_blank" rel="noopener noreferrer">${new URL(src).hostname}</a></li>`).join('');
            const verdictClass = data.verdict.toLowerCase().replace(/\s+/g, '-');

            container.innerHTML = `
                <div class="result-header">
                    <h3>Analysis for: ${data.product_name}</h3>
                </div>
                <div class="dashboard-grid">
                     <div class="grid-item glass-effect main-summary">
                        <div class="summary-box">
                            <div>
                                <h4>The Analyst's Verdict</h4>
                                <p>${data.summary}</p>
                                <span class="verdict ${verdictClass}" style="margin-top: 1rem; display: inline-block;">${data.verdict}</span>
                            </div>
                        </div>
                    </div>
                    <div class="grid-item glass-effect advantages">
                        <div class="advantages-box">
                            <h4><i data-lucide="thumbs-up"></i> Advantages</h4>
                            <ul>${advantagesHtml}</ul>
                        </div>
                    </div>
                     <div class="grid-item glass-effect disadvantages">
                        <div class="disadvantages-box">
                            <h4><i data-lucide="thumbs-down"></i> Disadvantages</h4>
                            <ul>${disadvantagesHtml}</ul>
                        </div>
                    </div>
                    <div class="grid-item glass-effect word-cloud">
                         <div class="viz-card">
                             <h4>Key Topic Cloud</h4>
                             <canvas id="wordcloud-canvas"></canvas>
                        </div>
                    </div>
                    <div class="grid-item glass-effect sentiment-chart">
                        <div class="viz-card">
                             <h4>Sentiment</h4>
                              <canvas id="sentimentChart"></canvas>
                        </div>
                    </div>
                     <div class="grid-item glass-effect sources">
                         <div class="sources-box">
                            <h4><i data-lucide="link"></i> Sources Analyzed</h4>
                            <ul>${sourcesHtml}</ul>
                        </div>
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
                list: list, gridSize: 10, weightFactor: 4, fontFamily: 'Inter, sans-serif',
                color: (word, weight) => {
                    const topicData = topics[word];
                    if (!topicData) return '#e5e7eb';
                    if (topicData.sentiment > 0.65) return '#10b981';
                    if (topicData.sentiment < 0.35) return '#ef4444';
                    return '#9ca3af';
                },
                backgroundColor: 'transparent', rotateRatio: 0, minSize: 10
            });
        },
        renderSentimentChart: (sentimentData) => {
            const ctx = document.getElementById('sentimentChart')?.getContext('2d');
            if (!ctx) return;
            if(sentimentChartInstance) sentimentChartInstance.destroy();
            sentimentChartInstance = new Chart(ctx, {
                type: 'doughnut', data: {
                    labels: [`Positive (${sentimentData.positive})`, `Negative (${sentimentData.negative})`],
                    datasets: [{ data: [sentimentData.positive, sentimentData.negative],
                        backgroundColor: [ 'rgba(16, 185, 129, 0.7)', 'rgba(239, 68, 68, 0.7)'],
                        borderColor: ['#10b981', '#ef4444'], borderWidth: 2, }]
                },
                options: { responsive: true, maintainAspectRatio: false, cutout: '70%',
                    plugins: { legend: { labels: { color: '#e5e7eb' } } }
                }
            });
        },
        renderComparisonResult: (container, data) => {
            const renderProductCard = (p) => `
                <div class="comparison-card glass-effect">
                    <h4>${p.product_name}</h4>
                    <span class="verdict ${p.verdict.toLowerCase().replace(/\s+/g, '-')}">${p.verdict}</span>
                    <div class="adv-disadv-comp">
                        <h5>Advantages</h5>
                        <ul>${p.advantages.map(adv => `<li><i data-lucide="thumbs-up"></i> ${adv}</li>`).join('')}</ul>
                    </div>
                    <div class="adv-disadv-comp">
                        <h5>Disadvantages</h5>
                         <ul>${p.disadvantages.map(dis => `<li><i data-lucide="thumbs-down"></i> ${dis}</li>`).join('')}</ul>
                    </div>
                </div>
            `;
            container.innerHTML = `
                <div class="summary-box glass-effect">
                     <i data-lucide="git-compare-arrows"></i>
                     <div>
                        <h4>The Final Verdict</h4>
                        <p>${data.comparison_summary.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')}</p>
                     </div>
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
        if (!query) { ui.showError(resultsContainer, 'Please enter a product name or URL.'); return; }
        ui.setButtonLoading(analyzeBtn, true);
        ui.showLoading(resultsContainer, `Analyzing "${query}"...`);
        try {
            const results = await handleFetch(`${API_BASE_URL}/api/analyze_product`, { query });
            ui.renderFullAnalysis(resultsContainer, results);
        } catch (error) { ui.showError(resultsContainer, error.message);
        } finally { ui.setButtonLoading(analyzeBtn, false); }
    };
    
    const handleComparison = async () => {
        if (!compareQuery1Input || !compareQuery2Input || !comparisonResultsContainer || !compareBtn) return;
        const query1 = compareQuery1Input.value.trim();
        const query2 = compareQuery2Input.value.trim();
        if (!query1 || !query2) { ui.showError(comparisonResultsContainer, 'Please provide two queries.'); return; }
        ui.setButtonLoading(compareBtn, true, 'Generate Comparison');
        ui.showLoading(comparisonResultsContainer, `Comparing products...`);
        try {
            const result = await handleFetch(`${API_BASE_URL}/api/compare_products`, { query1, query2 });
            ui.renderComparisonResult(comparisonResultsContainer, result);
        } catch (error) { ui.showError(comparisonResultsContainer, error.message);
        } finally { ui.setButtonLoading(compareBtn, false, 'Generate Comparison'); }
    };
    
    if (analyzeBtn) analyzeBtn.addEventListener('click', handleAnalysis);
    if (productQueryInput) productQueryInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') handleAnalysis();
    });
    if (compareBtn) compareBtn.addEventListener('click', handleComparison);

    lucide.createIcons();
});

