@echo off
echo Gerando executável com PyInstaller...
pyinstaller --noconfirm --onefile --console ^
 --add-data "templates;templates" ^
 --add-data "static;static" ^
 --add-data "model;model" ^
 --add-data "data;data" ^
 app.py

echo.
echo ✅ Executável gerado em: dist\app.exe
pause
