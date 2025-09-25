// This ensures the entire page is loaded before we try to add event listeners
document.addEventListener('DOMContentLoaded', () => {
    
    // --- Common Elements & API URL ---
    const API_BASE_URL = 'http://127.0.0.1:5000/api';

    // --- Page-specific Setup ---
    // This robust approach prevents errors by only running code for elements that exist on the current page.
    
    // Run this code ONLY if we are on the main dashboard (index.html)
    if (document.getElementById('productUrlInput')) {
        setupDashboardPage();
    }

    // Run this code ONLY if we are on the comparison page (compare.html)
    if (document.getElementById('compareBtn')) {
        setupComparePage();
    }

    // The bot is on every page, so we can set it up unconditionally
    setupSuggestionBot();

    // --- Dashboard Page Functions ---
    function setupDashboardPage() {
        const analyzeUrlBtn = document.getElementById('analyzeUrlBtn');
        const productUrlInput = document.getElementById('productUrlInput');
        const liveAnalysisResultEl = document.getElementById('liveAnalysisResult');
        const loadingSpinner = document.getElementById('loadingSpinner');
        const productListEl = document.getElementById('productList');
        const productDetailsEl = document.getElementById('productDetails');

        analyzeUrlBtn.addEventListener('click', async () => {
            const url = productUrlInput.value.trim();
            if (!url) {
                showError('Please paste a Flipkart product URL.', liveAnalysisResultEl);
                return;
            }
            
            loadingSpinner.classList.remove('hidden');
            liveAnalysisResultEl.classList.add('hidden');

            try {
                const response = await fetch(`${API_BASE_URL}/analyze_url`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ url })
                });
                const data = await handleResponse(response);
                renderProductDetails(data, liveAnalysisResultEl);
                liveAnalysisResultEl.classList.remove('hidden');
            } catch (error) {
                console.error('Analysis error:', error);
                showError(error.message, liveAnalysisResultEl);
                liveAnalysisResultEl.classList.remove('hidden');
            } finally {
                loadingSpinner.classList.add('hidden');
            }
        });

        const loadProducts = async () => {
            try {
                const response = await fetch(`${API_BASE_URL}/products`);
                const products = await handleResponse(response);
                productListEl.innerHTML = products.map(p => `<div class="product-item" data-id="${p.product_id}">${p.product_id}</div>`).join('');
                addProductListeners();
            } catch (error) {
                console.error('Error loading products:', error);
                productListEl.innerHTML = '<p class="error-text">Could not load products.</p>';
            }
        };

        const addProductListeners = () => {
            document.querySelectorAll('.product-item').forEach(item => {
                item.addEventListener('click', (e) => {
                    const productId = e.target.dataset.id;
                    loadProductDetails(productId, productDetailsEl);
                    document.querySelectorAll('.product-item').forEach(p => p.classList.remove('active'));
                    e.target.classList.add('active');
                });
            });
        };

        const loadProductDetails = async (productId, container) => {
             container.innerHTML = `<div class="spinner-container"><div class="spinner"></div></div>`;
            try {
                const response = await fetch(`${API_BASE_URL}/product/${productId}`);
                const data = await handleResponse(response);
                renderProductDetails(data, container);
            } catch (error) {
                console.error('Error loading product details:', error);
                container.innerHTML = '<p class="error-text">Could not load details.</p>';
            }
        };

        loadProducts();
    }

    // --- Comparison Page Functions ---
    function setupComparePage() {
        const compareBtn = document.getElementById('compareBtn');
        const productUrl1 = document.getElementById('productUrl1');
        const productUrl2 = document.getElementById('productUrl2');
        const comparisonResultEl = document.getElementById('comparisonResult');
        const loadingSpinnerCompare = document.getElementById('loadingSpinnerCompare');

        compareBtn.addEventListener('click', async () => {
            const url1 = productUrl1.value.trim();
            const url2 = productUrl2.value.trim();

            if (!url1 || !url2) {
                showError('Please paste two Flipkart product URLs to compare.', comparisonResultEl);
                comparisonResultEl.classList.remove('hidden');
                return;
            }

            loadingSpinnerCompare.classList.remove('hidden');
            comparisonResultEl.classList.add('hidden');

            try {
                const response = await fetch(`${API_BASE_URL}/compare_urls`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ url1, url2 })
                });
                const data = await handleResponse(response);
                renderComparison(data);
                comparisonResultEl.classList.remove('hidden');
            } catch (error) {
                console.error('Comparison error:', error);
                showError(error.message, comparisonResultEl);
                comparisonResultEl.classList.remove('hidden');
            } finally {
                loadingSpinnerCompare.classList.add('hidden');
            }
        });
    }

    // --- Suggestion Bot Functions ---
    function setupSuggestionBot() {
        const botToggle = document.getElementById('botToggle');
        const suggestionBot = document.getElementById('suggestionBot');
        const closeBot = document.getElementById('closeBot');
        const botBody = document.getElementById('botBody');
        const botInput = document.getElementById('botInput');
        const botSendBtn = document.getElementById('botSendBtn');

        botToggle.addEventListener('click', () => {
            suggestionBot.classList.toggle('open');
            botToggle.querySelector('[data-lucide="message-circle"]').classList.toggle('hidden');
            botToggle.querySelector('[data-lucide="x"]').classList.toggle('hidden');
        });

        closeBot.addEventListener('click', () => {
            suggestionBot.classList.remove('open');
            botToggle.querySelector('[data-lucide="message-circle"]').classList.remove('hidden');
            botToggle.querySelector('[data-lucide="x"]').classList.add('hidden');
        });
        
        const handleBotQuery = async () => {
            const query = botInput.value.trim();
            if (!query) return;

            addMessageToBot(query, 'user');
            botInput.value = '';

            try {
                const response = await fetch(`${API_BASE_URL}/recommend`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query })
                });
                const data = await handleResponse(response);
                addMessageToBot(data.recommendation, 'bot');
            } catch (error) {
                console.error('Bot error:', error);
                addMessageToBot('Sorry, I encountered an error. Please try again.', 'bot');
            }
        };

        botSendBtn.addEventListener('click', handleBotQuery);
        botInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') handleBotQuery();
        });
    }
    
    // --- Universal Helper & Rendering Functions ---
    let sentimentCharts = {}; // Store multiple chart instances

    function addMessageToBot(text, sender) {
        const botBody = document.getElementById('botBody');
        const messageEl = document.createElement('div');
        messageEl.classList.add('message', `${sender}-message`);
        let formattedText = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        formattedText = formattedText.replace(/(\n|^)\* (.*?)(?=\n\*|\n\n|$)/g, '$1<li>$2</li>');
        if (formattedText.includes('<li>')) {
            formattedText = `<ul>${formattedText.replace(/<\/li><li>/g, '</li><li>')}</ul>`;
        }
        messageEl.innerHTML = `<p>${formattedText.replace(/\n/g, '<br>')}</p>`;
        botBody.appendChild(messageEl);
        botBody.scrollTop = botBody.scrollHeight;
    }

    function renderProductDetails(data, container) {
        const verdictClass = data.verdict === 'Good' ? 'good' : 'bad';
        const verdictIcon = data.verdict === 'Good' ? 'thumbs-up' : 'thumbs-down';
        const chartId = `chart-${Math.random().toString(36).substr(2, 9)}`;

        container.innerHTML = `
            <div class="verdict ${verdictClass}">
                <i data-lucide="${verdictIcon}"></i>
                <div>
                    <strong>Verdict: A ${data.verdict} Pick</strong>
                    <p>Based on ${data.review_count} reviews</p>
                </div>
            </div>
            <div class="result-grid">
                <div class="chart-container">
                    <canvas id="${chartId}"></canvas>
                </div>
                <div class="summary-text">
                    <h3>AI-Generated Summary</h3>
                    <p>${data.summary}</p>
                </div>
            </div>
        `;
        renderChart(data.sentiment_distribution, chartId);
        lucide.createIcons();
    }
    
    function renderComparison(data) {
        const container = document.getElementById('comparisonResult');
        container.innerHTML = `
            <div class="comparison-summary card">
                <div class="card-header">
                    <i data-lucide="award"></i>
                    <h3>AI Recommendation</h3>
                </div>
                <p>${data.comparison_summary}</p>
            </div>
            <div class="comparison-grid">
                <div class="product-column card">${renderProductDetailsForCompare(data.product1, 'sentimentChart1')}</div>
                <div class="product-column card">${renderProductDetailsForCompare(data.product2, 'sentimentChart2')}</div>
            </div>
        `;
        renderChart(data.product1.sentiment_distribution, 'sentimentChart1');
        renderChart(data.product2.sentiment_distribution, 'sentimentChart2');
        lucide.createIcons();
    }

    function renderProductDetailsForCompare(data, canvasId) {
        const verdictClass = data.verdict === 'Good' ? 'good' : 'bad';
        const verdictIcon = data.verdict === 'Good' ? 'thumbs-up' : 'thumbs-down';
        return `
            <div class="card-header"><h4>Analysis</h4></div>
            <div class="verdict ${verdictClass}">
                 <i data-lucide="${verdictIcon}"></i>
                 <strong>${data.verdict} Pick</strong>
            </div>
             <p class="subtitle" style="margin-bottom: 1rem;">Based on ${data.review_count} reviews</p>
            <div class="chart-container">
                <canvas id="${canvasId}"></canvas>
            </div>
            <h5>AI Summary</h5>
            <p class="summary-text-small">${data.summary}</p>
        `;
    }

    function renderChart(distribution, canvasId) {
        if (!document.getElementById(canvasId)) return;
        const ctx = document.getElementById(canvasId).getContext('2d');
        const positiveCount = distribution.POSITIVE || 0;
        const negativeCount = distribution.NEGATIVE || 0;

        // Destroy old chart if it exists to prevent flickering
        if (sentimentCharts[canvasId]) {
            sentimentCharts[canvasId].destroy();
        }
        
        sentimentCharts[canvasId] = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Positive', 'Negative'],
                datasets: [{
                    label: 'Sentiment',
                    data: [positiveCount, negativeCount],
                    backgroundColor: ['#10b981', '#ef4444'],
                    borderColor: ['#ffffff'],
                    borderWidth: 3
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { font: { family: "'Inter', sans-serif", size: 14 }, color: '#6b7280', padding: 20 }
                    },
                    tooltip: {
                        titleFont: { family: "'Inter', sans-serif" },
                        bodyFont: { family: "'Inter', sans-serif" },
                    }
                },
                cutout: '70%'
            }
        });
    }

    async function handleResponse(response) {
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ message: 'The server returned an unexpected error.' }));
            const errorMessage = `Could not fetch reviews for this URL. The page might be protected or the layout has changed.`;
            throw new Error(errorMessage);
        }
        return response.json();
    }

    function showError(message, container) {
        container.innerHTML = `<div class="error-message">${message}</div>`;
        container.classList.remove('hidden');
    }
});

