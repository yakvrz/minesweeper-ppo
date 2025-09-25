const boardEl = document.getElementById('board');
const statusEl = document.getElementById('status');
const newGameBtn = document.getElementById('new-game');
const toggleOverlayBtn = document.getElementById('toggle-overlay');

let currentState = null;
let isBusy = false;
let overlayEnabled = true;

const updateOverlayControl = () => {
  toggleOverlayBtn.textContent = overlayEnabled ? 'Hide Overlay' : 'Show Overlay';
  toggleOverlayBtn.setAttribute('aria-pressed', overlayEnabled ? 'true' : 'false');
};

const formatPercent = (value) => {
  if (value === null || Number.isNaN(value)) {
    return '';
  }
  return `${Math.round(value * 100)}%`;
};

const probToColor = (value) => {
  if (value === null || Number.isNaN(value)) {
    return 'transparent';
  }
  const safe = Math.min(Math.max(value, 0), 1);
  const hue = safe * 120; // 0 -> red, 120 -> green
  return `hsla(${hue}, 70%, 50%, 0.65)`;
};

const updateStatus = (state, message = null) => {
  if (message) {
    statusEl.textContent = message;
    return;
  }
  if (!state) {
    statusEl.textContent = 'Loading…';
    return;
  }
  if (state.done) {
    if (state.outcome === 'win') {
      statusEl.textContent = `Win in ${state.step} steps`;
    } else if (state.outcome === 'loss') {
      statusEl.textContent = `Game over after ${state.step} steps`;
    } else {
      statusEl.textContent = 'Game finished';
    }
  } else {
    statusEl.textContent = `Step ${state.step}`;
  }
};

const renderBoard = (state) => {
  boardEl.classList.toggle('overlay-enabled', overlayEnabled);
  boardEl.style.setProperty('--rows', state.rows);
  boardEl.style.setProperty('--cols', state.cols);
  boardEl.innerHTML = '';

  for (let r = 0; r < state.rows; r += 1) {
    for (let c = 0; c < state.cols; c += 1) {
      const revealed = state.revealed[r][c];
      const prob = state.safe_probabilities[r][c];
      const count = state.counts[r][c];

      const cell = document.createElement('button');
      cell.type = 'button';
      cell.dataset.row = String(r);
      cell.dataset.col = String(c);
      cell.classList.add('cell');

      if (revealed) {
        cell.classList.add('revealed');
        if (count > 0) {
          cell.textContent = String(count);
          cell.classList.add(`count-${count}`);
        } else {
          cell.innerHTML = '&nbsp;';
        }
        cell.disabled = true;
      } else {
        cell.classList.add('hidden');
        cell.style.setProperty('--overlay-color', probToColor(prob));
        if (prob !== null) {
          cell.setAttribute('aria-label', `Cell (${r}, ${c}) safe probability ${Math.round(prob * 100)} percent`);
        }
        if (overlayEnabled && prob !== null) {
          const label = document.createElement('span');
          label.classList.add('prob-label');
          label.textContent = formatPercent(prob);
          cell.appendChild(label);
        }
        if (state.done) {
          cell.disabled = true;
        } else {
          cell.addEventListener('click', onCellClick);
        }
      }

      boardEl.appendChild(cell);
    }
  }
};

const handleResponse = (resp) => {
  if (!resp.ok) {
    throw new Error(`Request failed: ${resp.status}`);
  }
  return resp.json();
};

const fetchState = async () => {
  const resp = await fetch('/api/state');
  return handleResponse(resp);
};

const postJson = async (url, payload) => {
  const resp = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });
  return handleResponse(resp);
};

const refresh = (state) => {
  currentState = state;
  renderBoard(state);
  updateStatus(state);
};

const onCellClick = async (event) => {
  if (isBusy) {
    return;
  }
  const row = Number(event.currentTarget.dataset.row);
  const col = Number(event.currentTarget.dataset.col);

  isBusy = true;
  updateStatus(currentState, 'Revealing…');
  try {
    const nextState = await postJson('/api/click', { row, col });
    refresh(nextState);
  } catch (err) {
    console.error(err);
    updateStatus(currentState, 'Failed to reveal cell');
  } finally {
    isBusy = false;
  }
};

const startNewGame = async () => {
  if (isBusy) {
    return;
  }
  isBusy = true;
  updateStatus(null, 'Resetting…');
  try {
    const nextState = await postJson('/api/new-game', {});
    refresh(nextState);
  } catch (err) {
    console.error(err);
    updateStatus(currentState, 'Failed to reset game');
  } finally {
    isBusy = false;
  }
};

newGameBtn.addEventListener('click', startNewGame);
toggleOverlayBtn.addEventListener('click', () => {
  overlayEnabled = !overlayEnabled;
  updateOverlayControl();
  if (currentState) {
    renderBoard(currentState);
  }
});

const bootstrap = async () => {
  updateStatus(null, 'Loading…');
  updateOverlayControl();
  try {
    const state = await fetchState();
    refresh(state);
  } catch (err) {
    console.error(err);
    updateStatus(null, 'Failed to load state');
  }
};

bootstrap();
