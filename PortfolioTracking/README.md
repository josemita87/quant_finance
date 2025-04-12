# Portfolio Dashboard

An interactive dashboard for analyzing investment portfolio transactions with advanced visualizations and real-time updates.

## Features

- **Real-time Portfolio Analysis**: View key portfolio metrics updated in real-time
- **Interactive Charts**: Visualize portfolio evolution, returns, and transaction distributions
- **Transaction Management**: Filter and search through transaction history
- **Price Updates**: Update current price to see how it affects your portfolio returns
- **Responsive Design**: Works on desktop and mobile devices

## Project Structure

The project consists of two main components:

1. **Backend API** (Flask)
   - Provides portfolio analysis endpoints
   - Calculates metrics and returns
   - Generates visualization charts

2. **Frontend Dashboard** (Angular)
   - Interactive UI with Material Design
   - Real-time data synchronization
   - Responsive layout for all devices

## Setup Instructions

### Prerequisites

- Python 3.7+ and pip
- Node.js and npm
- Angular CLI

### Backend Setup

1. Set up a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the Flask API server:
   ```bash
   python api.py
   ```
   The API will be available at http://localhost:5000

### Frontend Setup

1. Navigate to the Angular project directory:
   ```bash
   cd portfolio-dashboard
   ```

2. Install the required npm packages:
   ```bash
   npm install
   ```

3. Start the Angular development server:
   ```bash
   ng serve
   ```
   The application will be available at http://localhost:4200

## Usage

1. Open the dashboard at http://localhost:4200
2. Navigate through the different sections using the sidebar
3. Use the "Update Current Price" form to see how different prices affect your returns
4. Explore your transaction history and filter by type or search for specific entries
5. View detailed metrics and performance charts

## API Endpoints

- `GET /api/metrics` - Get all portfolio metrics
- `GET /api/transactions` - Get transaction history
- `GET /api/charts/portfolio_evolution` - Get portfolio evolution chart
- `GET /api/charts/transaction_distribution` - Get transaction distribution chart
- `GET /api/charts/portfolio_returns` - Get portfolio returns chart
- `POST /api/price/update` - Update current price for analysis

## Technologies Used

- **Backend**:
  - Python
  - Flask
  - Pandas
  - Matplotlib
  - NumPy

- **Frontend**:
  - Angular
  - Angular Material
  - Chart.js
  - RxJS
  - TypeScript

## Author

Created by [Your Name] 