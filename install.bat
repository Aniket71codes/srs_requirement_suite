@echo off
echo ============================================
echo  DocuMind - One-time dependency install
echo ============================================
echo.

python -m pip install --user fastapi "uvicorn[standard]" python-multipart scikit-learn joblib pypdf python-docx numpy
python -m pip install --user torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install --user transformers tokenizers safetensors

echo.
echo ============================================
echo  Done! Run run.bat to start the project.
echo ============================================
pause
