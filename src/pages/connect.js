import { useState, useEffect, useRef, useCallback } from "react";
import Webcam from "react-webcam";
import * as faceapi from "face-api.js";

const WebcamComponent = () => {
  const [webcamRef, setWebcamRef] = useState(null);
  const [faces, setFaces] = useState([]);
  const [faceImgSrc, setFaceImgSrc] = useState([]);

  const detect = async () => {
    const video = webcamRef.video;
    const displaySize = { width: video.videoWidth, height: video.videoHeight };
    const detections = await faceapi.detectAllFaces(
      video,
      new faceapi.TinyFaceDetectorOptions()
    );
    const newFaces = [];
    const newFaceImgs = [];
    const faceImages = await Promise.all(
      detections.map(async (detection) => {
        if (!detection) return {};
        const faceCanvas = document.createElement("canvas");
        const faceContext = faceCanvas.getContext("2d");
        const faceImageData = faceContext.getImageData(
          detection.box.x,
          detection.box.y,
          detection.box.width,
          detection.box.height
        );
        const portionCanvas = document.createElement("canvas");
        portionCanvas.width = detection.box.width;
        portionCanvas.height = detection.box.height;
        const portionCtx = portionCanvas.getContext("2d");
        portionCtx.drawImage(
          video,
          detection.box.x - 50,
          detection.box.y - 100,
          detection.box.width + 100,
          detection.box.height + 100,
          0,
          0,
          detection.box.width,
          detection.box.height
        );
        const dataURLImage = portionCanvas.toDataURL("image/jpeg");
        if (!faces.some((f) => f.box.equals(detection.box))) {
          newFaces.push(detection);
          newFaceImgs.push(dataURLImage);
        }
        const faceTensor = await faceapi.tf.browser
          .fromPixels(faceImageData)
          .toFloat();
        return { tensor: faceTensor, src: dataURLImage };
      })
    );
    setFaces([...faces, ...newFaces]);
    setFaceImgSrc([...faceImgSrc, ...newFaceImgs]);
  };

  useEffect(() => {
    const loadModels = async () => {
      await faceapi.nets.tinyFaceDetector.loadFromUri("/models"),
        await faceapi.nets.faceRecognitionNet.loadFromUri("/models");
    };
    loadModels();
  }, []);

  useEffect(() => {
    if (webcamRef) {
      setInterval(() => {
        detect();
      }, 1000);
    }
  }, [webcamRef]);

  return (
    <div>
      <Webcam
        ref={(webcam) => setWebcamRef(webcam)}
        screenshotFormat="image/jpeg"
        style={{
          position: "absolute",
          marginLeft: "auto",
          marginRight: "auto",
          left: 0,
          right: 0,
          textAlign: "center",
          zIndex: 9,
          width: 640,
          height: 480,
        }}
      />
      {faceImgSrc &&
        faceImgSrc.map((imgSrc, i) => (
          <img
            key={i}
            src={imgSrc}
            width="250"
            height="250"
            style={{
              position: "absolute",
              top: 300 * i,
              left: 0,
              zIndex: 10,
            }}
          />
        ))}
    </div>
  );
};

export default WebcamComponent;
