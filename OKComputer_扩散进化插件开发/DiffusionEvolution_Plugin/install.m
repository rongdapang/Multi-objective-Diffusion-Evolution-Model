%% DiffusionEvolution Plugin Installation Script
% This script installs the DiffusionEvolution plugin into the PlatEMO framework.
%
% Usage:
%   install()                    % Interactive installation
%   install('platemo_path')      % Specify PlatEMO path
%   install('platemo_path', false) % Non-interactive installation
%
% The script will:
% 1. Verify PlatEMO installation
% 2. Copy plugin files to appropriate directories
% 3. Update MATLAB path
% 4. Run verification tests

function install(platemoPath, interactive)
    %% Parse input arguments
    if nargin < 1
        interactive = true;
        % Try to find PlatEMO automatically
        platemoPath = findPlatEMOPath();
        if isempty(platemoPath)
            if interactive
                platemoPath = input('Please enter the path to your PlatEMO installation: ', 's');
            else
                error('PlatEMO path not found. Please specify the path manually.');
            end
        end
    elseif nargin < 2
        interactive = true;
    end
    
    %% Step 1: Verify PlatEMO installation
    fprintf('Step 1: Verifying PlatEMO installation...\n');
    if ~verifyPlatEMO(platemoPath)
        error('PlatEMO not found at %s. Please check the path.', platemoPath);
    end
    fprintf('  ✓ PlatEMO found at: %s\n', platemoPath);
    
    %% Step 2: Create target directory
    targetDir = fullfile(platemoPath, 'PlatEMO', 'Algorithms', 'Multi-objective optimization', 'DiffusionEvolution');
    fprintf('Step 2: Creating target directory...\n');
    if exist(targetDir, 'dir')
        if interactive
            response = input(sprintf('Target directory already exists:\n  %s\nOverwrite? (y/N): ', targetDir), 's');
            if ~strcmpi(response, 'y')
                fprintf('Installation cancelled.\n');
                return;
            end
        end
        rmdir(targetDir, 's');
    end
    mkdir(targetDir);
    fprintf('  ✓ Created directory: %s\n', targetDir);
    
    %% Step 3: Copy plugin files
    fprintf('Step 3: Copying plugin files...\n');
    sourceDir = fileparts(mfilename('fullpath'));
    
    % Copy algorithm files
    copyfile(fullfile(sourceDir, 'DiffusionEvolution', '*.m'), targetDir);
    fprintf('  ✓ Copied algorithm files\n');
    
    %% Step 4: Update MATLAB path
    fprintf('Step 4: Updating MATLAB path...\n');
    addpath(targetDir);
    savepath();
    fprintf('  ✓ Added to MATLAB path\n');
    
    %% Step 5: Run verification tests
    fprintf('Step 5: Running verification tests...\n');
    if runVerificationTests()
        fprintf('  ✓ All tests passed\n');
    else
        warning('Some tests failed. Please check the implementation.');
    end
    
    %% Installation complete
    fprintf('\n=== Installation Complete ===\n');
    fprintf('DiffusionEvolution has been successfully installed!\n\n');
    fprintf('You can now use the algorithm in your PlatEMO projects:\n');
    fprintf('  Algorithm = DiffusionEvolution();\n');
    fprintf('  Algorithm.Solve(Problem);\n\n');
    
    fprintf('For examples and documentation, see:\n');
    fprintf('  %s\n', fullfile(sourceDir, 'examples', 'example_usage.m'));
    fprintf('  %s\n', fullfile(sourceDir, 'README.md'));
    
end

function platemoPath = findPlatEMOPath()
    % Try to find PlatEMO automatically
    platemoPath = '';
    
    % Common installation paths
    commonPaths = {
        'C:\PlatEMO',
        'D:\PlatEMO',
        fullfile(getenv('USERPROFILE'), 'Documents', 'MATLAB', 'PlatEMO'),
        fullfile(getenv('HOME'), 'Documents', 'MATLAB', 'PlatEMO'),
        '/usr/local/PlatEMO',
        '/opt/PlatEMO'
    };
    
    for i = 1:length(commonPaths)
        if exist(commonPaths{i}, 'dir')
            if exist(fullfile(commonPaths{i}, 'platemo.m'), 'file')
                platemoPath = commonPaths{i};
                return;
            end
        end
    end
    
    % Search in MATLAB path
    matlabPath = strsplit(path, pathsep);
    for i = 1:length(matlabPath)
        if exist(fullfile(matlabPath{i}, 'platemo.m'), 'file')
            platemoPath = matlabPath{i};
            return;
        end
    end
end

function isValid = verifyPlatEMO(platemoPath)
    % Verify PlatEMO installation
    isValid = false;
    
    % Check if path exists
    if ~exist(platemoPath, 'dir')
        return;
    end
    
    % Check for key files
    keyFiles = {
        'platemo.m',
        'PlatEMO/Algorithms/ALGORITHM.m',
        'PlatEMO/Problems/PROBLEM.m'
    };
    
    for i = 1:length(keyFiles)
        if ~exist(fullfile(platemoPath, keyFiles{i}), 'file')
            return;
        end
    end
    
    isValid = true;
end

function success = runVerificationTests()
    % Run basic verification tests
    success = true;
    
    fprintf('  Running basic tests...\n');
    
    % Test 1: Algorithm instantiation
    try
        Algorithm = DiffusionEvolution();
        fprintf('    ✓ Algorithm instantiation\n');
    catch ME
        fprintf('    ✗ Algorithm instantiation failed: %s\n', ME.message);
        success = false;
    end
    
    % Test 2: Parameter configuration
    try
        Algorithm = DiffusionEvolution('parameter', {50, 100, 20, 0.3, 'DDPM', 5, 'linear', 'fitness'});
        fprintf('    ✓ Parameter configuration\n');
    catch ME
        fprintf('    ✗ Parameter configuration failed: %s\n', ME.message);
        success = false;
    end
    
    % Test 3: Problem creation
    try
        Problem = ZDT1();
        fprintf('    ✓ Problem creation\n');
    catch ME
        fprintf('    ✗ Problem creation failed: %s\n', ME.message);
        success = false;
    end
    
    % Test 4: Short optimization run
    try
        Problem = ZDT1();
        Algorithm = DiffusionEvolution('parameter', {20, 50, 10, 0.3, 'DDPM', 2, 'linear', 'none'});
        Algorithm.Solve(Problem);
        fprintf('    ✓ Short optimization run\n');
    catch ME
        fprintf('    ✗ Short optimization run failed: %s\n', ME.message);
        success = false;
    end
    
end