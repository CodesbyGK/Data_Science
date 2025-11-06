function showTab(tabId) {
  document.querySelectorAll('.tab').forEach(tab => {
    tab.classList.remove('active');
  });
  document.getElementById(tabId).classList.add('active');
}

//db connector
document.getElementById('predictForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const teamA = e.target.teamA.value;
  const teamB = e.target.teamB.value;
  const lineup = e.target.lineup.value;

  const response = await fetch('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ team_a: teamA, team_b: teamB, lineup })
  });

  const result = await response.json();
  document.getElementById('predictionResult').innerText =
    `Prediction: ${result.prediction} (${result.probability}%)`;
});