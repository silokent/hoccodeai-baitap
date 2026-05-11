const fs = require("fs");
const readline = require("readline");
const URL = "http://127.0.0.1:7860";

function base64ToImage(base64String, savePath = "output_image.png") {
  const buffer = Buffer.from(base64String, "base64");
  fs.writeFileSync(savePath, buffer);
}

function ask(question) {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  return new Promise((resolve) =>
    rl.question(question, (ans) => {
      rl.close();
      resolve(ans);
    })
  );
}

async function textToImage(promptInput, width, height) {
  const payload = {
    prompt: promptInput,
    negative_prompt:
      "worst quality, low quality, watermark, text, error, blurry, jpeg artifacts, cropped, jpeg artifacts, signature, watermark, username, artist name, bad anatomy",
    steps: 25,
    cfg_scale: 7.5,
    width: Number(width),
    height: Number(width),
  };

  try {
    console.log("Sending Inference Request");
    const response = await fetch(`${URL}/sdapi/v1/txt2img`, {
      method: "POST",
      body: JSON.stringify(payload),
      headers: { "Content-Type": "application/json" },
    });
    const respJson = await response.json();
    console.log(respJson);
    console.log("Inference Completed");
    respJson.images.forEach((img, index) => {
      console.log(`Saving image output_image_${index}.png`);
      base64ToImage(img, `output_image_${index}.png`);
    });
  } catch (error) {
    console.error("Error:", error);
  }
}

(async () => {
  const promptInput = await ask("Enter prompt: ");
  const width = await ask("Enter width: ");
  const height = await ask("Enter height: ");

  await textToImage(promptInput, width, height);
})();