setlocal

:: https://github.com/smroid/cedar-solve.git
:: https://github.com/smroid/cedar-solve/tree/master/tetra3

set repository_url=https://github.com/smroid/cedar-solve.git
set repository2_url=https://github.com/smroid/cedar-detect.git
set subdirectory_name=tetra3
set temp_folder=cedar_temp
set desired_destination_folder=cedar_solve
set desired_destination2_folder=cedar_detect

:: goto STEP2
:STEP1
echo STEP1:
echo Cloning %repository_url% to %temp_folder%
rmdir /s /q %temp_folder%
rmdir /s /q %desired_destination_folder%

git clone --no-checkout %repository_url% %temp_folder%
git clone %repository2_url% %desired_destination2_folder%
pause

:STEP2
echo STEP2:
echo CD to: %temp_folder%
cd %temp_folder%

echo Sparse CHECKOUT %subdirectory_name%

git sparse-checkout init
git sparse-checkout set %subdirectory_name%
git checkout
cd ..

:STEP3
echo STEP3
echo move %temp_folder%/%subdirectory_name% to %desired_destination_folder%
pause
move %temp_folder%/%subdirectory_name% %desired_destination_folder%

echo Creating python __init__.py files:
type nul > %desired_destination_folder%\__init__.py
type nul > %desired_destination2_folder%\python\__init__.py

::rmdir /s /q temp_directory
pause

endlocal