const https = require('https');
const fs = require('fs');
const path = require('path');

// Function to download an image
function downloadImage(url, filepath) {
    return new Promise((resolve, reject) => {
        const file = fs.createWriteStream(filepath);
        
        https.get(url, (response) => {
            if (response.statusCode !== 200) {
                reject(new Error(`Failed to download: ${response.statusCode}`));
                return;
            }
            
            response.pipe(file);
            
            file.on('finish', () => {
                file.close();
                resolve();
            });
            
            file.on('error', (err) => {
                fs.unlink(filepath, () => {}); // Delete the file if there was an error
                reject(err);
            });
        }).on('error', (err) => {
            reject(err);
        });
    });
}

// Function to get filename from URL
function getFilenameFromURL(url) {
    const urlParts = url.split('/');
    const filename = urlParts[urlParts.length - 2];
    const extension = urlParts[urlParts.length - 1].split('.')[1];
    const newFilename = `${filename}.${extension}`;
    return newFilename;
}

// Function to generate random delay between 1-3 seconds
function getRandomDelay() {
    return Math.floor(Math.random() * 2000) + 1000; // 1000-3000ms
}

// Main function to download all images
async function downloadAllImages() {
    try {
        // Read the JSON file
        const jsonData = JSON.parse(fs.readFileSync('elodea-images.json', 'utf8'));
        const images = jsonData.images;
        // Trim images array to 333 images
        const maxImages = 333;
        if (images.length > maxImages) {
            images.length = maxImages;
        }
        
        // Create target directory if it doesn't exist
        const targetDir = '../dataset/train/other';
        if (!fs.existsSync(targetDir)) {
            fs.mkdirSync(targetDir, { recursive: true });
        }
        
        console.log(`Starting download of ${images.length} images...`);
        console.log(`Target directory: ${targetDir}`);
        
        let downloadedCount = 0;
        let skippedCount = 0;
        let errorCount = 0;
        
        for (let i = 0; i < images.length; i++) {
            const imageUrl = images[i];
            const filename = getFilenameFromURL(imageUrl);
            const filepath = path.join(targetDir, filename);
            
            // Check if file already exists
            if (fs.existsSync(filepath)) {
                console.log(`[${i + 1}/${images.length}] Skipping ${filename} - already exists`);
                skippedCount++;
                continue;
            }
            
            try {
                console.log(`[${i + 1}/${images.length}] Downloading ${filename}...`);
                await downloadImage(imageUrl, filepath);
                downloadedCount++;
                console.log(`✓ Downloaded ${filename}`);
                
                // Add random delay between downloads
                const delay = getRandomDelay();
                console.log(`Waiting ${delay}ms before next download...`);
                await new Promise(resolve => setTimeout(resolve, delay));
                
            } catch (error) {
                console.error(`✗ Error downloading ${filename}: ${error.message}`);
                errorCount++;
            }
        }
        
        console.log('\n=== Download Summary ===');
        console.log(`Total images: ${images.length}`);
        console.log(`Downloaded: ${downloadedCount}`);
        console.log(`Skipped (already exists): ${skippedCount}`);
        console.log(`Errors: ${errorCount}`);
        console.log('========================');
        
    } catch (error) {
        console.error('Error reading JSON file:', error.message);
    }
}

// Run the downloader
downloadAllImages().catch(console.error);
