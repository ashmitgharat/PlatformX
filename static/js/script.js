document.getElementById("startBtn").addEventListener("click", () => {
    fetch("/start", { method: "POST" })
        .then(response => response.text())
        .then(data => alert(data));
});

document.getElementById("stopBtn").addEventListener("click", () => {
    fetch("/stop", { method: "POST" })
        .then(response => response.text())
        .then(data => {
            alert(data);
            document.getElementById("downloadBtn").style.display = "block";
        });
});
