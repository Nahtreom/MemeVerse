let dialogsData = [];
let currentDialogIndex = 0;
let dialogIds = [];

function renderDialog(index) {
  const chat = document.getElementById('chat');
  chat.innerHTML = '';
  if (!dialogsData || dialogsData.length === 0) {
    chat.innerHTML = '<div style="text-align:center;color:#888;">No dialog content available</div>';
    return;
  }
  if (index < 0 || index >= dialogsData.length) return;
  const dialog = dialogsData[index];
  const fullDialog = dialog.full_dialog;
  // Page title uses dialog_id
  const titleDiv = document.createElement('div');
  titleDiv.style.textAlign = 'center';
  titleDiv.style.fontWeight = 'bold';
  titleDiv.style.margin = '10px 0 20px 0';
  titleDiv.textContent = dialog.dialog_id || (dialogIds[index] || '');
  chat.appendChild(titleDiv);
  for (let i = 0; i < fullDialog.length; i++) {
    let line = fullDialog[i];
    let role = 'assistant';
    if (line.startsWith('A:')) {
      role = 'user';
      line = line.replace(/^A:/, '').trim();
    } else if (line.startsWith('B:')) {
      line = line.replace(/^B:/, '').trim();
    }
    // Check if it's an image
    const imgMatch = line.match(/^(\S+\.(?:png|jpg|jpeg))$/i);
    if (imgMatch) {
      // Only display the image, not the original text
      const imgDiv = document.createElement('div');
      imgDiv.className = 'message ' + role;
      const img = document.createElement('img');
      img.src = '../Meme Warehouse/EmojoPackage_processed/' + encodeURIComponent(imgMatch[1]);
      img.alt = 'Meme Image';
      img.style.maxWidth = '120px';
      img.style.display = 'block';
      img.style.margin = '8px 0';
      imgDiv.appendChild(img);
      chat.appendChild(imgDiv);
    } else {
      // Normal text
      const msgDiv = document.createElement('div');
      msgDiv.className = 'message ' + role;
      const bubble = document.createElement('div');
      bubble.className = 'bubble';
      bubble.textContent = line;
      msgDiv.appendChild(bubble);
      chat.appendChild(msgDiv);
    }
  }
  // Add navigation buttons
  const navDiv = document.createElement('div');
  navDiv.style.textAlign = 'center';
  navDiv.style.marginTop = '20px';
  const prevBtn = document.createElement('button');
  prevBtn.textContent = 'Previous';
  prevBtn.disabled = (index === 0);
  prevBtn.onclick = function() {
    if (currentDialogIndex > 0) {
      currentDialogIndex--;
      renderDialog(currentDialogIndex);
    }
  };
  const nextBtn = document.createElement('button');
  nextBtn.textContent = 'Next';
  nextBtn.disabled = (index === dialogsData.length - 1);
  nextBtn.onclick = function() {
    if (currentDialogIndex < dialogsData.length - 1) {
      currentDialogIndex++;
      renderDialog(currentDialogIndex);
    }
  };
  navDiv.appendChild(prevBtn);
  navDiv.appendChild(document.createTextNode('  '));
  navDiv.appendChild(nextBtn);
  chat.appendChild(navDiv);
}
// change the resultFileName to the name of the json file you want to load
let resultFileName='role_based_18_turns_Diversity-awareSelection.json'
fetch('../Dialogs_with_meme/'+resultFileName)
  .then(res => {
    if (!res.ok) {
      throw new Error('Failed to load '+resultFileName);
    }
    return res.json();
  })
  .then(dialogs => {
    dialogsData = dialogs;
    dialogIds = dialogs.map(d => d.dialog_id);
    currentDialogIndex = 0;
    renderDialog(currentDialogIndex);
  })
  .catch(err => {
    const chat = document.getElementById('chat');
    chat.innerHTML = '<div style="text-align:center;color:red;">Failed to load dialog: ' + err.message + '</div>';
  });
