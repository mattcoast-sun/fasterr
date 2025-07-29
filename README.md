# 🪙⚡ Bitcoin Energy Analyzer for watsonx Orchestrate

An AI-powered FastAPI service designed specifically for **watsonx Orchestrate** that analyzes the correlation between Bitcoin price movements and US energy consumption patterns. This tool provides structured insights and visualizations for automation workflows and business intelligence.

## 🎯 watsonx Orchestrate Integration

This API is optimized for watsonx Orchestrate with:
- **Structured API responses** with Pydantic models
- **Comprehensive error handling** for reliable automation
- **CORS support** for cross-origin requests
- **OpenAPI/Swagger documentation** for easy skill creation
- **AI-generated insights** for business intelligence workflows

## 📊 Key Features

### Business Intelligence
- **Real-time Bitcoin price analysis** from CoinGecko API
- **Energy consumption correlation modeling** with mining estimates
- **Statistical insights generation** with correlation coefficients
- **Automated reporting** with structured data outputs
- **Visual chart generation** for presentations and dashboards

### Technical Capabilities
- **RESTful API** with comprehensive documentation
- **Pydantic data validation** ensuring reliable data flows
- **Fallback data sources** for high availability
- **Professional visualizations** with dual-axis charts
- **Cloud-native deployment** ready for IBM Code Engine

## 🔗 Main API Endpoint for watsonx Orchestrate

### `POST /analyze-bitcoin-energy`

**Primary endpoint for watsonx Orchestrate automation workflows**

#### Request Schema:
```json
{
  "days": 180,
  "include_correlation": true,
  "format": "base64"
}
```

#### Response Schema:
```json
{
  "success": true,
  "image_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
  "correlation_coefficient": 0.742,
  "summary": "Analysis of 180 days from 2024-01-01 to 2024-06-30. Correlation coefficient: 0.742. Bitcoin ranged from $41,234 to $67,891.",
  "period_analyzed": "2024-01-01 to 2024-06-30",
  "bitcoin_price_range": {
    "min": 41234.56,
    "max": 67891.23
  },
  "energy_usage_range": {
    "min": 312.45,
    "max": 367.89
  },
  "insights": [
    "Bitcoin price increased 23.4% over the analyzed period",
    "Strong positive correlation detected between Bitcoin price and energy usage",
    "Above-average energy consumption period, possibly due to increased mining activity"
  ]
}
```

## 🚀 watsonx Orchestrate Use Cases

### 1. **Financial Market Monitoring**
- Automated Bitcoin price trend analysis
- Energy market impact assessments
- Correlation-based trading signals
- Risk management insights

### 2. **Energy Infrastructure Planning**
- Bitcoin mining energy demand forecasting
- Grid capacity planning support
- Renewable energy allocation optimization
- Environmental impact reporting

### 3. **Business Intelligence Dashboards**
- Automated report generation
- Executive summary creation
- Market trend visualization
- Stakeholder communication

### 4. **Regulatory Compliance**
- Energy consumption reporting
- Environmental impact documentation
- Market volatility analysis
- Risk assessment automation

## 🛠️ watsonx Orchestrate Skill Creation

### Step 1: Import API
1. In watsonx Orchestrate, go to **Skills → Create Skill**
2. Choose **Import from OpenAPI**
3. Use URL: `https://your-app-url/docs` (OpenAPI JSON available at `/openapi.json`)

### Step 2: Configure Skill
- **Skill Name**: "Bitcoin Energy Analyzer"
- **Description**: "Analyzes Bitcoin price correlation with US energy consumption"
- **Primary Endpoint**: `/analyze-bitcoin-energy`

### Step 3: Skill Parameters
```yaml
Parameters:
  - days: "Number of days to analyze (30-365)"
  - include_correlation: "Include statistical correlation"
  - format: "Response format preference"

Output:
  - Chart image (base64)
  - Correlation insights
  - Price range analysis
  - Energy consumption data
  - AI-generated summaries
```

## 📈 Business Value Propositions

### For Energy Companies
- **Demand Forecasting**: Predict energy demand spikes from Bitcoin mining
- **Infrastructure Planning**: Plan grid capacity for crypto mining operations
- **Market Analysis**: Understand crypto market impact on energy markets

### For Financial Institutions
- **Risk Assessment**: Analyze Bitcoin volatility correlation with energy costs
- **Investment Decisions**: Factor energy consumption into crypto investment strategies
- **Regulatory Reporting**: Generate compliance reports for energy-intensive assets

### For Government Agencies
- **Policy Making**: Data-driven cryptocurrency regulation
- **Environmental Monitoring**: Track energy consumption patterns
- **Economic Analysis**: Understand crypto market impact on national energy grid

## 🔧 Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

**Access Points:**
- **API Documentation**: http://localhost:8080/docs
- **Gradio UI**: http://localhost:8080/gradio
- **Health Check**: http://localhost:8080/health
- **Main Analysis**: `POST http://localhost:8080/analyze-bitcoin-energy`

## ☁️ IBM Code Engine Deployment

### Quick Deploy for watsonx Orchestrate

1. **Source Code Deployment** (Recommended):
   ```bash
   # Create Code Engine application
   ibmcloud ce application create \
     --name bitcoin-energy-analyzer \
     --build-source https://github.com/your-repo \
     --build-strategy buildpacks \
     --port 8080
   ```

2. **Container Deployment**:
   ```bash
   # Build and push
   docker build -t bitcoin-energy-analyzer .
   docker tag bitcoin-energy-analyzer icr.io/namespace/bitcoin-energy-analyzer
   docker push icr.io/namespace/bitcoin-energy-analyzer
   
   # Deploy to Code Engine
   ibmcloud ce application create \
     --name bitcoin-energy-analyzer \
     --image icr.io/namespace/bitcoin-energy-analyzer \
     --port 8080
   ```

### Environment Configuration
```bash
# Required for production
PORT=8080
CORS_ORIGINS=https://your-watsonx-orchestrate-domain.ibm.com
API_RATE_LIMIT=100
```

## 📋 API Reference

### All Endpoints

| Endpoint | Method | Purpose | watsonx Orchestrate Ready |
|----------|--------|---------|---------------------------|
| `/analyze-bitcoin-energy` | POST | Main analysis endpoint | ✅ Primary |
| `/health` | GET | Health monitoring | ✅ Status checks |
| `/` | GET | API information | ✅ Discovery |
| `/docs` | GET | OpenAPI documentation | ✅ Skill creation |
| `/gradio` | GET | Interactive UI | ⚠️ Demo only |

### Error Handling
- **400**: Invalid input parameters
- **500**: Analysis processing errors
- **503**: External API unavailable (fallback data provided)

### Rate Limiting
- Default: 100 requests per minute
- Configurable via environment variables
- Graceful degradation to cached/mock data

## 🔒 Security Considerations

### For Production Deployment
```bash
# Recommended environment variables
ALLOWED_ORIGINS=https://watsonx-orchestrate.ibm.com
API_KEY_REQUIRED=true
LOG_LEVEL=INFO
RATE_LIMIT_ENABLED=true
```

### CORS Configuration
- Pre-configured for IBM watsonx Orchestrate domains
- Customizable via environment variables
- Supports enterprise security requirements

## 📊 Monitoring & Analytics

### Health Checks
- **Endpoint**: `/health`
- **Response Time**: < 100ms
- **Availability**: 99.9% target
- **Dependencies**: CoinGecko API status

### Performance Metrics
- **Average Response Time**: ~2-3 seconds
- **Chart Generation**: ~1-2 seconds
- **Data Processing**: ~0.5 seconds
- **API Call Overhead**: ~0.1 seconds

## 🤝 Support & Documentation

- **API Documentation**: Available at `/docs` endpoint
- **Interactive Testing**: Available at `/gradio` endpoint
- **GitHub Issues**: For bug reports and feature requests
- **watsonx Orchestrate Community**: For integration support

---

**Ready for watsonx Orchestrate!** 🚀 This API provides comprehensive Bitcoin-energy correlation analysis with enterprise-grade reliability and structured outputs perfect for AI-powered automation workflows. 