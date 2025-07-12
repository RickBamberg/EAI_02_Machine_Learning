@echo off
echo ============================================
echo  Executando script para salvar modelo e pre-processador...
echo ============================================
echo.

REM Muda o diretorio atual para o diretorio ONDE ESTE SCRIPT (.bat) ESTA LOCALIZADO.
REM Isso garante que os caminhos relativos (venv, salva_modelo.py) funcionem
REM independentemente de onde voce execute o .bat.
cd /D "%~dp0"
echo Diretorio atual definido para: %cd%
echo.

REM Ativa o ambiente virtual (procura por venv\Scripts\activate.bat)
if exist "venv\Scripts\activate.bat" (
    echo Ativando ambiente virtual 'venv'...
    call "venv\Scripts\activate.bat"
    echo Ambiente virtual ativado.
    echo.
) else (
    echo Aviso: Ambiente virtual 'venv' nao encontrado ou estrutura nao padrao.
    echo Tentando executar com o Python do sistema/PATH...
    echo.
)

REM Executa o script Python
echo Executando: python salva_modelo.py
echo --------------------------------------------
python salva_modelo.py
echo --------------------------------------------
echo.

REM Verifica se o script Python executou com sucesso (opcional, mas bom)
if %errorlevel% neq 0 (
    echo ERRO: O script Python 'salva_modelo.py' retornou um erro (codigo: %errorlevel%).
    echo Verifique a saida acima para detalhes.
) else (
    echo SUCESSO: O script Python 'salva_modelo.py' foi concluido.
)
echo.

echo ============================================
echo  Processo concluido. Pressione qualquer tecla para sair.
echo ============================================
pause > nul
