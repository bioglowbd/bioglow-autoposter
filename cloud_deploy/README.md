# BioGlow AutoPoster - Cloud Deployment Guide (24/7 Always Online)

এই ফোল্ডারের ফাইলগুলো দিয়ে আপনি **Render.com** (সম্পূর্ণ ফ্রি ক্লাউড হোস্টিং)-এ ডেপ্লয় করতে পারবেন। এর ফলে আপনার কম্পিউটার বন্ধ থাকলেও মোবাইল অ্যাপ ২৪ ঘণ্টা, ৩৬৫ দিন সারা জীবন ইন্টারনেটে সচল থাকবে!

---

### ৩ মিনিটে ফ্রি ক্লাউড ডেপ্লয়মেন্টের সহজ ধাপ:

1. **GitHub-এ একটি ফ্রি রিপোজিটরি তৈরি করুন:**
   - [github.com](https://github.com)-এ যান।
   - একটি নতুন Private বা Public রিপোজিটরি তৈরি করুন (যেমন: `bioglow-autoposter`).
   - এই `cloud_deploy` ফোল্ডারের সবগুলো ফাইল সেখানে আপলোড (Push/Upload) করুন।

2. **Render.com-এ কানেক্ট করুন:**
   - [render.com](https://render.com)-এ গিয়ে একটি ফ্রি একাউন্ট খুলুন।
   - **"New +"** বাটনে ক্লিক করে **"Web Service"** সিলেক্ট করুন।
   - আপনার GitHub রিপোজিটরি (`bioglow-autoposter`) সিলেক্ট করুন।

3. **Environment Variables দিন:**
   Render-এর Environment সেকশনে নিচের ৫টি কী ও মান বসিয়ে দিন (যা আপনার লোকাল `.env` ফাইলে আছে):
   - `FACEBOOK_PAGE_ID`
   - `PAGE_ACCESS_TOKEN`
   - `INSTAGRAM_ACCOUNT_ID`
   - `GEMINI_API_KEY`
   - `OPENAI_API_KEY` (যদি ChatGPT ব্যবহার করতে চান)

4. **"Deploy Web Service"** বাটনে চাপ দিন!
   - ২ মিনিটের মধ্যে Render আপনাকে একটি স্থায়ী লিংক দিয়ে দেবে, যেমন:
     👉 `https://bioglow-autoposter.onrender.com`
   - এই লিংকটি কখনোই পরিবর্তন হবে না এবং কম্পিউটার বন্ধ থাকলেও মোবাইল থেকে সারা জীবন চলবে!
