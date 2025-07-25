echo "Setting up AI Article Assistant..."

# Create backend conda environment with local prefix
echo "Creating conda environment..."
cd backend
conda create --prefix ./backend-env python=3.11 -y
conda activate ./backend-env

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Setup frontend
echo "Setting up frontend..."
cd ..
npm install

echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit backend/.env with your Azure OpenAI credentials"
echo "2. Activate the conda environment: conda activate ./backend/backend-env"
echo "3. Start the backend (from project root): uvicorn backend.main:app --reload"
echo "4. In a new terminal, start the frontend: npm run dev"
echo "5. Open http://localhost:3000 in your browser"
echo ""
echo "Note: The conda environment was created but needs to be manually activated in your terminal."
