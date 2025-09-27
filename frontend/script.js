// ENHANCED FRONTEND SCRIPT - FIXED WORD CLOUD & COMPARISON ISSUES
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
    
    // Chart instances for cleanup
    let sentimentChartInstance = null;
    let wordCloudInstance = null;

    // --- Enhanced API Fetch Function ---
    async function handleFetch(url, body, retries = 3) {
        for (let i = 0; i < retries; i++) {
            try {
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), 30000); // 30s timeout
                
                const response = await fetch(url, {
                    method: 'POST',
                    headers: { 
                        'Content-Type': 'application/json',
                        'Accept': 'application/json'
                    },
                    body: JSON.stringify(body),
                    signal: controller.signal
                });
                
                clearTimeout(timeoutId);
                
                const data = await response.json();
                if (!response.ok) {
                    throw new Error(data.error || `Server error: ${response.status} ${response.statusText}`);
                }
                return data;
            } catch (error) {
                if (error.name === 'AbortError') {
                    throw new Error('Request timed out. Please try again.');
                }
                if (i === retries - 1) throw error;
                await new Promise(resolve => setTimeout(resolve, 1000 * (i + 1))); // Exponential backoff
            }
        }
    }
    
    // --- Enhanced UI Rendering ---
    const ui = {
        setButtonLoading: (button, isLoading, originalText = '') => {
            if (!button) return;
            
            if (isLoading) {
                button.disabled = true;
                button.innerHTML = '<div class="spinner-small"></div>';
                button.classList.add('loading');
            } else {
                button.disabled = false;
                if (originalText && !originalText.includes('<')) {
                    button.innerHTML = `<i data-lucide="${originalText}"></i>`;
                } else if (originalText) {
                    button.innerHTML = originalText;
                } else {
                    button.innerHTML = '<i data-lucide="arrow-right"></i>';
                }
                button.classList.remove('loading');
                lucide.createIcons();
            }
        },
        
        showLoading: (container, message) => {
            if (!container) return;
            container.innerHTML = `
                <div class="loading-indicator glass-effect">
                    <div class="spinner"></div>
                    <p>${message}</p>
                    <small style="color: var(--text-muted);">This may take 30-60 seconds...</small>
                </div>`;
            container.classList.remove('hidden');
        },
        
        showError: (container, message, suggestion = '') => {
            if (!container) return;
            container.innerHTML = `
                <div class="loading-indicator glass-effect" style="border-left: 4px solid var(--accent-red);">
                    <i data-lucide="alert-triangle" style="color: var(--accent-red); width: 48px; height: 48px;"></i>
                    <h3 style="color: var(--accent-red); margin: 0;">Error</h3>
                    <p>${message}</p>
                    ${suggestion ? `<small style="color: var(--text-muted); margin-top: 1rem;">${suggestion}</small>` : ''}
                    <button onclick="location.reload()" style="margin-top: 1rem; padding: 0.5rem 1rem; background: var(--accent-blue); color: white; border: none; border-radius: 6px; cursor: pointer;">
                        Refresh Page
                    </button>
                </div>`;
            container.classList.remove('hidden');
            lucide.createIcons();
        },
        
        renderFullAnalysis: (container, data) => {
            // Cleanup previous instances
            if (sentimentChartInstance) {
                sentimentChartInstance.destroy();
                sentimentChartInstance = null;
            }
            
            const advantagesHtml = (data.advantages || []).map((adv, index) => 
                `<li data-aos="fade-up" data-aos-delay="${index * 100}">
                    <i data-lucide="check-circle-2"></i> 
                    <span>${adv}</span>
                </li>`
            ).join('');
            
            const disadvantagesHtml = (data.disadvantages || []).map((dis, index) => 
                `<li data-aos="fade-up" data-aos-delay="${index * 100}">
                    <i data-lucide="x-circle"></i> 
                    <span>${dis}</span>
                </li>`
            ).join('');
            
            const sourcesHtml = (data.sources || []).map((src, index) => {
                try {
                    const hostname = new URL(src).hostname.replace('www.', '');
                    return `<li><a href="${src}" target="_blank" rel="noopener noreferrer">${hostname}</a></li>`;
                } catch {
                    return `<li><a href="${src}" target="_blank" rel="noopener noreferrer">Source ${index + 1}</a></li>`;
                }
            }).join('');
            
            const verdictClass = (data.verdict || 'mixed-opinions').toLowerCase().replace(/\s+/g, '-');
            const verdictIcon = this.getVerdictIcon(data.verdict);

            container.innerHTML = `
                <div class="result-header">
                    <h3>Analysis Results</h3>
                    <p style="color: var(--text-secondary); font-size: 1.1rem;">${data.product_name || 'Product Analysis'}</p>
                </div>
                
                <div class="dashboard-grid">
                    <div class="grid-item glass-effect main-summary">
                        <div class="summary-box">
                            <h4><i data-lucide="brain-circuit"></i> AI Analysis</h4>
                            <p>${data.summary || 'Analysis completed successfully.'}</p>
                            <div style="display: flex; align-items: center; gap: 1rem; margin-top: 1rem;">
                                <span class="verdict ${verdictClass}">
                                    ${verdictIcon} ${data.verdict || 'Mixed Opinions'}
                                </span>
                                ${this.renderConfidenceScore(data)}
                            </div>
                        </div>
                    </div>
                    
                    <div class="grid-item glass-effect advantages">
                        <div class="advantages-box">
                            <h4><i data-lucide="thumbs-up"></i> Strengths (${(data.advantages || []).length})</h4>
                            <ul>${advantagesHtml || '<li><i data-lucide="info"></i> <span>No specific advantages identified</span></li>'}</ul>
                        </div>
                    </div>
                    
                    <div class="grid-item glass-effect disadvantages">
                        <div class="disadvantages-box">
                            <h4><i data-lucide="thumbs-down"></i> Weaknesses (${(data.disadvantages || []).length})</h4>
                            <ul>${disadvantagesHtml || '<li><i data-lucide="info"></i> <span>No specific disadvantages identified</span></li>'}</ul>
                        </div>
                    </div>
                    
                    <div class="grid-item glass-effect sentiment-chart">
                        <div class="viz-card">
                            <h4><i data-lucide="pie-chart"></i> Sentiment Overview</h4>
                            <div style="position: relative; height: 250px;">
                                <canvas id="sentimentChart"></canvas>
                            </div>
                            ${this.renderSentimentStats(data.sentiment)}
                        </div>
                    </div>
                    
                    <div class="grid-item glass-effect word-cloud">
                        <div class="viz-card">
                            <h4><i data-lucide="cloud"></i> Key Topics</h4>
                            <div style="position: relative; height: 250px; background: rgba(15, 23, 42, 0.5); border-radius: 8px; overflow: hidden;">
                                <canvas id="wordcloud-canvas"></canvas>
                            </div>
                            <small style="color: var(--text-muted); margin-top: 0.5rem; display: block;">
                                <span style="color: var(--accent-green);">●</span> Positive 
                                <span style="color: var(--accent-red); margin-left: 1rem;">●</span> Negative 
                                <span style="color: var(--text-secondary); margin-left: 1rem;">●</span> Neutral
                            </small>
                        </div>
                    </div>
                    
                    ${sourcesHtml ? `
                    <div class="grid-item glass-effect sources">
                        <div class="sources-box">
                            <h4><i data-lucide="link"></i> Sources Analyzed (${data.sources.length})</h4>
                            <ul>${sourcesHtml}</ul>
                        </div>
                    </div>` : ''}
                </div>`;
            
            container.classList.remove('hidden');
            
            // Render visualizations with delay to ensure DOM is ready
            setTimeout(() => {
                this.renderEnhancedWordCloud(data.key_topics);
                this.renderSentimentChart(data.sentiment);
                lucide.createIcons();
            }, 100);
        },
        
        renderEnhancedWordCloud: (topics) => {
            const canvas = document.getElementById('wordcloud-canvas');
            if (!canvas || !topics || Object.keys(topics).length === 0) {
                // Show fallback if no topics
                if (canvas) {
                    const ctx = canvas.getContext('2d');
                    canvas.width = canvas.offsetWidth;
                    canvas.height = canvas.offsetHeight;
                    ctx.fillStyle = 'var(--text-muted)';
                    ctx.font = '16px Inter';
                    ctx.textAlign = 'center';
                    ctx.fillText('No key topics identified', canvas.width/2, canvas.height/2);
                }
                return;
            }

            try {
                // Prepare word list with enhanced formatting
                const wordList = Object.entries(topics)
                    .sort(([,a], [,b]) => b.count - a.count)
                    .slice(0, 50) // Limit to top 50 words
                    .map(([word, data]) => [
                        word.charAt(0).toUpperCase() + word.slice(1), // Capitalize first letter
                        Math.max(data.count * 15, 12) // Scale size appropriately
                    ]);

                // Configure canvas
                canvas.width = canvas.offsetWidth;
                canvas.height = canvas.offsetHeight;
                
                // Enhanced WordCloud configuration
                WordCloud(canvas, {
                    list: wordList,
                    gridSize: 8,
                    weightFactor: function(size) {
                        return Math.pow(size, 0.7) * 0.8;
                    },
                    fontFamily: 'Inter, sans-serif',
                    fontWeight: '600',
                    color: (word, weight) => {
                        const originalWord = word.toLowerCase();
                        const topicData = topics[originalWord];
                        if (!topicData) return '#94a3b8';
                        
                        if (topicData.sentiment > 0.7) return '#10b981';
                        if (topicData.sentiment < 0.3) return '#ef4444';
                        if (topicData.sentiment > 0.5) return '#06b6d4';
                        return '#94a3b8';
                    },
                    backgroundColor: 'transparent',
                    rotateRatio: 0.3,
                    rotationSteps: 2,
                    minSize: 12,
                    drawOutOfBound: false,
                    shrinkToFit: true
                });
            } catch (error) {
                console.error('WordCloud error:', error);
                // Fallback display
                const ctx = canvas.getContext('2d');
                ctx.fillStyle = '#ef4444';
                ctx.font = '14px Inter';
                ctx.textAlign = 'center';
                ctx.fillText('Word cloud unavailable', canvas.width/2, canvas.height/2);
            }
        },
        
        renderSentimentChart: (sentimentData) => {
            const ctx = document.getElementById('sentimentChart')?.getContext('2d');
            if (!ctx || !sentimentData) return;
            
            // Cleanup previous instance
            if (sentimentChartInstance) {
                sentimentChartInstance.destroy();
                sentimentChartInstance = null;
            }
            
            const positive = sentimentData.positive || 0;
            const negative = sentimentData.negative || 0;
            const total = positive + negative;
            
            if (total === 0) {
                // Show empty state
                ctx.fillStyle = '#64748b';
                ctx.font = '14px Inter';
                ctx.textAlign = 'center';
                ctx.fillText('No sentiment data', ctx.canvas.width/2, ctx.canvas.height/2);
                return;
            }
            
            sentimentChartInstance = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: [`Positive (${positive})`, `Negative (${negative})`],
                    datasets: [{
                        data: [positive, negative],
                        backgroundColor: [
                            'rgba(16, 185, 129, 0.8)',
                            'rgba(239, 68, 68, 0.8)'
                        ],
                        borderColor: [
                            '#10b981',
                            '#ef4444'
                        ],
                        borderWidth: 3,
                        hoverBackgroundColor: [
                            'rgba(16, 185, 129, 0.9)',
                            'rgba(239, 68, 68, 0.9)'
                        ]
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    cutout: '65%',
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: {
                                color: '#f1f5f9',
                                font: {
                                    family: 'Inter',
                                    size: 12
                                },
                                padding: 20,
                                usePointStyle: true,
                                pointStyle: 'circle'
                            }
                        },
                        tooltip: {
                            backgroundColor: 'rgba(30, 41, 59, 0.9)',
                            titleColor: '#f1f5f9',
                            bodyColor: '#f1f5f9',
                            borderColor: 'rgba(148, 163, 184, 0.2)',
                            borderWidth: 1,
                            cornerRadius: 8
                        }
                    },
                    animation: {
                        animateScale: true,
                        duration: 1000
                    }
                }
            });
        },
        
        renderSentimentStats: (sentimentData) => {
            if (!sentimentData) return '';
            
            const positive = sentimentData.positive || 0;
            const negative = sentimentData.negative || 0;
            const total = positive + negative;
            
            if (total === 0) return '<p style="text-align: center; color: var(--text-muted);">No sentiment data available</p>';
            
            const positivePercent = ((positive / total) * 100).toFixed(1);
            const negativePercent = ((negative / total) * 100).toFixed(1);
            
            return `
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem; font-size: 0.9rem;">
                    <div style="text-align: center;">
                        <div style="color: var(--accent-green); font-weight: 600; font-size: 1.5rem;">${positivePercent}%</div>
                        <div style="color: var(--text-muted);">Positive</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="color: var(--accent-red); font-weight: 600; font-size: 1.5rem;">${negativePercent}%</div>
                        <div style="color: var(--text-muted);">Negative</div>
                    </div>
                </div>`;
        },
        
        getVerdictIcon: (verdict) => {
            if (!verdict) return '🤔';
            const v = verdict.toLowerCase();
            if (v.includes('good') || v.includes('buy')) return '✅';
            if (v.includes('not') || v.includes('avoid')) return '❌';
            if (v.includes('consider') || v.includes('alternatives')) return '⚠️';
            return '🤔';
        },
        
        renderConfidenceScore: (data) => {
            const total = (data.sentiment?.positive || 0) + (data.sentiment?.negative || 0);
            let confidence = 'Low';
            let color = 'var(--accent-red)';
            
            if (total > 50) {
                confidence = 'High';
                color = 'var(--accent-green)';
            } else if (total > 20) {
                confidence = 'Medium';
                color = 'var(--accent-orange)';
            }
            
            return `
                <div style="display: flex; align-items: center; gap: 0.5rem; font-size: 0.9rem;">
                    <span style="color: var(--text-muted);">Confidence:</span>
                    <span style="color: ${color}; font-weight: 600;">${confidence}</span>
                    <span style="color: var(--text-muted);">(${total} reviews)</span>
                </div>`;
        },
        
        renderComparisonResult: (container, data) => {
            const renderProductCard = (product, index) => `
                <div class="comparison-card glass-effect" style="animation-delay: ${index * 0.2}s;">
                    <h4>${product.product_name}</h4>
                    <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1.5rem;">
                        <span class="verdict ${(product.verdict || 'mixed-opinions').toLowerCase().replace(/\s+/g, '-')}">
                            ${this.getVerdictIcon(product.verdict)} ${product.verdict || 'Mixed Opinions'}
                        </span>
                        ${this.renderConfidenceScore(product)}
                    </div>
                    
                    <div class="adv-disadv-comp">
                        <h5><i data-lucide="thumbs-up" style="color: var(--accent-green);"></i> Strengths</h5>
                        <ul>
                            ${(product.advantages || []).map(adv => 
                                `<li><i data-lucide="check-circle-2" style="color: var(--accent-green);"></i> <span>${adv}</span></li>`
                            ).join('') || '<li><i data-lucide="info"></i> <span>No specific advantages identified</span></li>'}
                        </ul>
                    </div>
                    
                    <div class="adv-disadv-comp">
                        <h5><i data-lucide="thumbs-down" style="color: var(--accent-red);"></i> Weaknesses</h5>
                        <ul>
                            ${(product.disadvantages || []).map(dis => 
                                `<li><i data-lucide="x-circle" style="color: var(--accent-red);"></i> <span>${dis}</span></li>`
                            ).join('') || '<li><i data-lucide="info"></i> <span>No specific disadvantages identified</span></li>'}
                        </ul>
                    </div>
                </div>
            `;
            
            container.innerHTML = `
                <div class="result-header">
                    <h3>Product Comparison</h3>
                    <p style="color: var(--text-secondary); font-size: 1.1rem;">AI-powered side-by-side analysis</p>
                </div>
                
                <div class="summary-box glass-effect" style="margin-bottom: 2rem;">
                    <h4><i data-lucide="brain-circuit"></i> Final Verdict</h4>
                    <div style="background: var(--glass-bg); padding: 1.5rem; border-radius: 8px; border-left: 4px solid var(--accent-blue);">
                        <p style="font-size: 1.1rem; line-height: 1.7; margin: 0;">
                            ${(data.comparison_summary || 'Comparison completed successfully.').replace(/\*\*(.*?)\*\*/g, '<strong style="color: var(--accent-blue);">$1</strong>')}
                        </p>
                    </div>
                </div>
                
                <div class="comparison-grid">
                    ${renderProductCard(data.product1 || {}, 0)}
                    ${renderProductCard(data.product2 || {}, 1)}
                </div>
            `;
            
            container.classList.remove('hidden');
            setTimeout(() => lucide.createIcons(), 100);
        }
    };

    // --- Enhanced Event Handlers ---
    const handleAnalysis = async () => {
        if (!productQueryInput || !resultsContainer || !analyzeBtn) return;
        
        const query = productQueryInput.value.trim();
        if (!query) { 
            ui.showError(resultsContainer, 'Please enter a product name or URL to analyze.', 'Try something like "iPhone 15 Pro" or "Sony WH-1000XM4"'); 
            return; 
        }
        
        ui.setButtonLoading(analyzeBtn, true);
        ui.showLoading(resultsContainer, `Analyzing "${query}"...`);
        
        try {
            const results = await handleFetch(`${API_BASE_URL}/api/analyze_product`, { query });
            ui.renderFullAnalysis(resultsContainer, results);
        } catch (error) { 
            console.error('Analysis error:', error);
            ui.showError(resultsContainer, error.message, 'Check your internet connection and try again.');
        } finally { 
            ui.setButtonLoading(analyzeBtn, false); 
        }
    };
    
    const handleComparison = async () => {
        if (!compareQuery1Input || !compareQuery2Input || !comparisonResultsContainer || !compareBtn) return;
        
        const query1 = compareQuery1Input.value.trim();
        const query2 = compareQuery2Input.value.trim();
        
        if (!query1 || !query2) { 
            ui.showError(comparisonResultsContainer, 'Please provide two products to compare.', 'Example: "iPhone 15" vs "Samsung Galaxy S24"'); 
            return; 
        }
        
        const originalText = 'Generate Comparison';
        ui.setButtonLoading(compareBtn, true);
        ui.showLoading(comparisonResultsContainer, `Comparing "${query1}" vs "${query2}"...`);
        
        try {
            const result = await handleFetch(`${API_BASE_URL}/api/compare_products`, { query1, query2 });
            ui.renderComparisonResult(comparisonResultsContainer, result);
        } catch (error) { 
            console.error('Comparison error:', error);
            ui.showError(comparisonResultsContainer, error.message, 'Make sure both product names are valid and try again.');
        } finally { 
            ui.setButtonLoading(compareBtn, false, originalText);
        }
    };
    
    // --- Event Listeners ---
    if (analyzeBtn) analyzeBtn.addEventListener('click', handleAnalysis);
    if (productQueryInput) {
        productQueryInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') handleAnalysis();
        });
    }
    if (compareBtn) compareBtn.addEventListener('click', handleComparison);
    
    // Initialize icons
    lucide.createIcons();
    
    // Add some example suggestions
    if (productQueryInput) {
        const suggestions = [
            'iPhone 15 Pro',
            'Sony WH-1000XM4',
            'Tesla Model 3',
            'MacBook Pro M3',
            'Samsung Galaxy S24'
        ];
        
        productQueryInput.addEventListener('focus', function() {
            if (!this.value) {
                this.placeholder = suggestions[Math.floor(Math.random() * suggestions.length)];
            }
        });
    }
});