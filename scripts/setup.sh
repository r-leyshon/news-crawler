echo "Setting up AI Article Assistant..."

# Create backend virtual environment
echo "Creating Python virtual environment..."
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Create ChromaDB directory
mkdir -p chroma_db

# Copy environment file
cp .env.example .env
echo "Please edit backend/.env with your Azure OpenAI credentials"

# Setup frontend
echo "Setting up frontend..."
cd ..
npm install

echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit backend/.env with your Azure OpenAI credentials"
echo "2. Start the backend: cd backend && uvicorn main:app --reload"
echo "3. Start the frontend: npm run dev"
echo "4. Open http://localhost:3000 in your browser"
