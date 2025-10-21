const fs = require('fs');
const path = require('path');

// Function to randomly shuffle an array (Fisher-Yates algorithm)
function shuffleArray(array) {
    const shuffled = [...array];
    for (let i = shuffled.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    return shuffled;
}

// Function to move files randomly to test folder
function moveToTest() {
    
    try {
        const trainDir = '../dataset/train/other';
        const testDir = '../dataset/val/other';
        
        // Check if directories exist
        if (!fs.existsSync(trainDir)) {
            console.error(`Train directory not found: ${trainDir}`);
            return;
        }
        
        // Create test directory if it doesn't exist
        if (!fs.existsSync(testDir)) {
            fs.mkdirSync(testDir, { recursive: true });
            console.log(`Created test directory: ${testDir}`);
        }
        
        // Get all files from train directory
        const trainFiles = fs.readdirSync(trainDir).filter(file => {
            const filePath = path.join(trainDir, file);
            return fs.statSync(filePath).isFile();
        });
        
        console.log(`Found ${trainFiles.length} files in train directory`);
        
        if (trainFiles.length < 100) {
            console.error(`Not enough files in train directory. Found: ${trainFiles.length}, Need: 100`);
            return;
        }
        
        // Randomly shuffle the files and take 100 for test
        console.log('\n=== Moving files to TEST ===');
        const shuffledFiles = shuffleArray(trainFiles);
        const filesForTest = shuffledFiles.slice(0, 100);
        
        let movedToTest = 0;
        let errorCount = 0;
        
        filesForTest.forEach((filename, index) => {
            const sourcePath = path.join(trainDir, filename);
            const destPath = path.join(testDir, filename);
            
            try {
                fs.renameSync(sourcePath, destPath);
                movedToTest++;
                console.log(`[${index + 1}/100] Moved to TEST: ${filename}`);
            } catch (error) {
                console.error(`Error moving ${filename} to test: ${error.message}`);
                errorCount++;
            }
        });
        
        console.log(`✓ Moved ${movedToTest} files to test folder`);
        
        // Final summary
        console.log('\n=== Final Summary ===');
        console.log(`Files moved to test: ${movedToTest}`);
        console.log(`Errors: ${errorCount}`);
        console.log(`Remaining in train: ${trainFiles.length - movedToTest}`);
        console.log('====================');
        
    } catch (error) {
        console.error('Error:', error.message);
    }
}

// Run the script
moveToTest();
