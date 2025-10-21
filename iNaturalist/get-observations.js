const https = require('https');
const fs = require('fs');

// Function to fetch JSON data from a URL
function fetchJSON(url) {
    return new Promise((resolve, reject) => {
        https.get(url, (res) => {
            let data = '';
            
            res.on('data', (chunk) => {
                data += chunk;
            });
            
            res.on('end', () => {
                try {
                    const jsonData = JSON.parse(data);
                    resolve(jsonData);
                } catch (error) {
                    reject(new Error(`Failed to parse JSON: ${error.message}`));
                }
            });
        }).on('error', (err) => {
            reject(err);
        });
    });
}

// Function to process image URL - replace square.jpg with large.jpg
function processImageURL(url) {
    if (!url) return null;
    return url.replace(/square\.jpg$/, 'large.jpg').replace(/square\.jpeg$/, 'large.jpeg');
}

// Function to process a single page
async function processPage(pageNum) {
    const url = `https://api.inaturalist.org/v2/observations?page=${pageNum}&place_id=1&taxon_id=57624&fields=(photos:(id:!t,url:!t))`;
    
    try {
        console.log(`Fetching page ${pageNum}...`);
        const data = await fetchJSON(url);
        
        // Check if there are results
        if (!data.results || data.results.length === 0) {
            console.log(`No results found on page ${pageNum}. Stopping.`);
            return []; // Return empty array to signal no more data
        }
        
        console.log(`Found ${data.results.length} observations on page ${pageNum}`);
        
        const imageURLs = [];
        
        // Process each observation
        data.results.forEach((observation, obsIndex) => {
            if (observation.photos && observation.photos.length > 0) {
                observation.photos.forEach((photo, photoIndex) => {
                    if (photo.url) {
                        const processedURL = processImageURL(photo.url);
                        if (processedURL) {
                            imageURLs.push(processedURL);
                            console.log(`Page ${pageNum}, Observation ${obsIndex + 1}, Photo ${photoIndex + 1}: ${processedURL}`);
                        }
                    }
                });
            }
        });
        
        return imageURLs;
        
    } catch (error) {
        console.error(`Error processing page ${pageNum}:`, error.message);
        return []; // Return empty array on error
    }
}

// Main function to loop through pages and collect all image URLs
async function scrapeObservations() {
    let pageNum = 1;
    let allImageURLs = [];
    let hasMoreData = true;
    
    console.log('Starting to scrape iNaturalist observations from API...');
    
    while (hasMoreData) {
        const pageImageURLs = await processPage(pageNum);
        
        if (pageImageURLs.length === 0) {
            hasMoreData = false;
        } else {
            allImageURLs = allImageURLs.concat(pageImageURLs);
            
            // Wait 1 second before next request
            console.log('Waiting 1 second before next request...');
            await new Promise(resolve => setTimeout(resolve, 1000));
            pageNum++;
        }
    }
    
    // Write results to JSON file
    const outputFile = 'elodea-images.json';
    const outputData = {
        totalImages: allImageURLs.length,
        images: allImageURLs,
        scrapedAt: new Date().toISOString()
    };
    
    try {
        fs.writeFileSync(outputFile, JSON.stringify(outputData, null, 2));
        console.log(`\nScraping completed successfully!`);
        console.log(`Total images collected: ${allImageURLs.length}`);
        console.log(`Results saved to: ${outputFile}`);
    } catch (error) {
        console.error(`Error writing to file: ${error.message}`);
    }
}

// Run the scraper
scrapeObservations().catch(console.error);
