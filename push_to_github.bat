@echo off
title Push Solar AI Project to GitHub
echo =====================================================================
echo  PUSHING SOLAR POWER ANOMALY DETECTION PROJECT TO GITHUB
echo =====================================================================
echo.
echo Remote URL: https://github.com/Soukhyaa/solar-power-anomaly-detection.git
echo Branch: main
echo.
git push -u origin main
echo.
if %errorlevel% neq 0 (
    echo.
    echo -----------------------------------------------------------------
    echo If it failed with "Repository not found":
    echo 1. Go to https://github.com/new
    echo 2. Repository name: solar-power-anomaly-detection
    echo 3. Keep it Public and do NOT check "Add README"
    echo 4. Click "Create repository"
    echo 5. Run this file again!
    echo -----------------------------------------------------------------
) else (
    echo Project successfully uploaded to GitHub!
)
pause
