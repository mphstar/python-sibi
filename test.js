function preProcessLandmark(landmarkList) {
  // Deep copy the landmark list
  let tempLandmarkList = JSON.parse(JSON.stringify(landmarkList));

  // Convert to relative coordinates
  let baseX = 0,
    baseY = 0;
  for (let i = 0; i < tempLandmarkList.length; i++) {
    const landmarkPoint = tempLandmarkList[i];
    if (i === 0) {
      baseX = landmarkPoint[0];
      baseY = landmarkPoint[1];
    }
    tempLandmarkList[i][0] -= baseX;
    tempLandmarkList[i][1] -= baseY;
  }

  // Convert to a one-dimensional list
  tempLandmarkList = [].concat(...tempLandmarkList);

  // Normalization
  const maxValue = Math.max(...tempLandmarkList.map(Math.abs));

  // Normalize function
  const normalize = (n) => n / maxValue;

  // Apply normalization
  tempLandmarkList = tempLandmarkList.map(normalize);

  return tempLandmarkList;
}

console.log(
  preProcessLandmark([
    [1, 2],
    [3, 4],
    [5, 6],
  ])
);
