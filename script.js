const video = document.createElement("video");
video.setAttribute("autoplay", true);
document.body.append(video);

const captureButton = document.createElement("button");
captureButton.textContent = "Capture Photo";
document.body.append(captureButton);

navigator.mediaDevices
  .getUserMedia({ video: {} })
  .then((stream) => {
    video.srcObject = stream;
  })
  .catch((error) => {
    console.error("Error accessing webcam:", error);
  });

Promise.all([
  faceapi.nets.faceRecognitionNet.loadFromUri("/models"),
  faceapi.nets.faceLandmark68Net.loadFromUri("/models"),
  faceapi.nets.ssdMobilenetv1.loadFromUri("/models"),
]).then(start);

async function start() {
  const container = document.createElement("div");
  container.style.position = "relative";
  document.body.append(container);
  const labeledFaceDescriptors = await loadLabeledImages();
  const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6);

  let canvas;
  document.body.append("Loaded");

  video.addEventListener("play", async () => {
    canvas = faceapi.createCanvasFromMedia(video);
    container.append(canvas);

    const displaySize = { width: video.width, height: video.height };
    faceapi.matchDimensions(canvas, displaySize);

    captureButton.addEventListener("click", () => {
      // Pause the video to capture a frame
      video.pause();

      // Capture the frame as an image
      const photo = faceapi.createCanvasFromMedia(video);
      const photoContext = photo.getContext("2d");
      photoContext.drawImage(video, 0, 0, video.width, video.height);

      // Resume the video
      video.play();

      // Process the captured photo
      processPhoto(photo);
    });

    setInterval(() => {
      // Trigger capture every 5 seconds (adjust as needed)
      captureButton.click();
    }, 5000);
  });
}

function processPhoto(photo) {
  const container = document.createElement("div");
  container.style.position = "relative";
  document.body.append(container);

  container.append(photo);

  const canvas = faceapi.createCanvasFromMedia(photo);
  container.append(canvas);

  const displaySize = { width: photo.width, height: photo.height };
  faceapi.matchDimensions(canvas, displaySize);

  const detections = faceapi
    .detectAllFaces(photo)
    .withFaceLandmarks()
    .withFaceDescriptors();

  const resizedDetections = faceapi.resizeResults(detections, displaySize);

  const results = resizedDetections.map((d) =>
    faceMatcher.findBestMatch(d.descriptor)
  );

  results.forEach((result, i) => {
    const box = resizedDetections[i].detection.box;
    const drawBox = new faceapi.draw.DrawBox(box, {
      label: result.toString(),
    });
    drawBox.draw(canvas);
  });

  // Remove webcam and photo elements after processing
  container.remove();
}

function loadLabeledImages() {
  const labels = ["Sehaj", "Parker", "Shubh"];
  return Promise.all(
    labels.map(async (label) => {
      const descriptions = [];
      for (let i = 1; i <= 2; i++) {
        const img = await faceapi.fetchImage(
          `https://raw.githubusercontent.com/PariPTP/AIAttendance/master/labeled_images/${label}/${i}.jpg`
        );
        const detections = await faceapi
          .detectSingleFace(img)
          .withFaceLandmarks()
          .withFaceDescriptor();
        descriptions.push(detections.descriptor);
      }

      return new faceapi.LabeledFaceDescriptors(label, descriptions);
    })
  );
}
