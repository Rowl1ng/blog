const fs = require('fs');
const path = require('path');
const axios = require('axios');
const FormData = require('form-data');

const smmsToken = 'AqXTeE7kFWSX4IXnQnYuufjR4Ko2FgZ6'; // 替换为你的 SM.MS Token
const markdownFile = '/Users/luoling/Documents/projects/blog/_posts/write/2014-4-4-Goldbach-conjecture.md'; // 或目录遍历处理多文件

// 下载图片到临时文件
async function downloadImage(url, filename) {
    const res = await axios.get(url, { responseType: 'arraybuffer' });
    const tempPath = path.join(__dirname, filename);
    fs.writeFileSync(tempPath, res.data);
    return tempPath;
}

// 上传到 SM.MS
async function uploadToSmms(filePath) {
    const form = new FormData();
    form.append('smfile', fs.createReadStream(filePath));
    const res = await axios.post('https://sm.ms/api/v2/upload', form, {
        headers: {
            ...form.getHeaders(),
            Authorization: smmsToken
        }
    });
    if (res.data.success) {
        return res.data.data.url;
    } else {
        console.log('上传失败：', res.data.message);
        return null;
    }
}

// 主处理函数
async function processMarkdown(filePath) {
    let content = fs.readFileSync(filePath, 'utf8');
    
    // 匹配类似：[1]: http://static.zybuluo.com/...
    const regex = /^\s*(\[\d+\]:\s+)(http:\/\/static\.zybuluo\.com\/.+)$/gm;
    const matches = [...content.matchAll(regex)];

    for (const match of matches) {
        const prefix = match[1]; // [编号]:
        const url = match[2];    // 原 URL
        console.log('正在处理:', url);
        
        try {
            const filename = path.basename(url);
            const tempPath = await downloadImage(url, filename);
            const newUrl = await uploadToSmms(tempPath);
            fs.unlinkSync(tempPath); // 删除临时文件

            if (newUrl) {
                content = content.replace(url, newUrl);
                console.log('替换为:', newUrl);
            }
        } catch (err) {
            console.log('处理失败:', url, err.message);
        }
    }

    fs.writeFileSync(filePath, content, 'utf8');
    console.log('处理完成:', filePath);
}

// 执行
processMarkdown(markdownFile);
