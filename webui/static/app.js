const boardEl = document.getElementById('board');
const statusEl = document.getElementById('status');
const newGameBtn = document.getElementById('new-game');
const toggleOverlayBtn = document.getElementById('toggle-overlay');
const boardSelect = document.getElementById('board-select');
const gameOverEl = document.getElementById('game-over');
const gameOverMessageEl = document.getElementById('game-over-message');
const gameOverResetBtn = document.getElementById('game-over-reset');

let currentState = null;
let isBusy = false;
let overlayEnabled = true;
let presetCacheSignature = '';

const updateOverlayControl = () => {
  toggleOverlayBtn.textContent = overlayEnabled ? 'Hide Mine Probabilities' : 'Show Mine Probabilities';
  toggleOverlayBtn.setAttribute('aria-pressed', overlayEnabled ? 'true' : 'false');
};

const formatPercent = (value) => {
  if (value === null || Number.isNaN(value)) {
    return '';
  }
  return `${Math.round(value * 100)}%`;
};

const mineColor = (prob) => {
  const mine = Math.min(Math.max(prob, 0), 1);
  const alpha = mine * 0.85;
  return `rgba(239, 68, 68, ${alpha})`;
};

const applyMineProbabilityDecor = (cell, overlayEl, labelEl, safeProb, row, col) => {
  const rowIdx = row + 1;
  const colIdx = col + 1;
  const hasProb = safeProb !== null && !Number.isNaN(safeProb);
  if (!hasProb) {
    cell.classList.remove('show-overlay');
    overlayEl.style.backgroundColor = 'transparent';
    overlayEl.style.opacity = 0;
    labelEl.textContent = '';
    labelEl.setAttribute('aria-hidden', 'true');
    cell.removeAttribute('data-mine-percent');
    cell.setAttribute('title', `Cell ${rowIdx}-${colIdx}: mine probability unavailable`);
    cell.setAttribute('aria-label', `Cell ${rowIdx}, ${colIdx}: mine probability unavailable`);
    cell.style.removeProperty('--cell-border');
    return;
  }

  const mineProb = 1 - safeProb;
  const clamped = Math.min(Math.max(mineProb, 0), 1);
  const percent = Math.round(clamped * 100);
  overlayEl.style.backgroundColor = mineColor(clamped);
  const overlayActive = overlayEnabled && clamped > 0;
  const overlayStrength = overlayActive ? Math.min(1, clamped + 0.05) : 0;
  overlayEl.style.opacity = overlayStrength;
  labelEl.textContent = overlayActive ? `${percent}%` : '';
  labelEl.setAttribute('aria-hidden', overlayActive ? 'false' : 'true');
  cell.dataset.minePercent = String(percent);
  cell.setAttribute('title', `Cell ${rowIdx}-${colIdx}: mine probability ${percent}%`);
  cell.setAttribute('aria-label', `Cell ${rowIdx}, ${colIdx}: mine probability ${percent} percent`);

  if (overlayActive) {
    cell.classList.add('show-overlay');
    cell.style.setProperty('--cell-border', `rgba(239, 68, 68, ${clamped * 0.6})`);
  } else {
    cell.classList.remove('show-overlay');
    cell.style.removeProperty('--cell-border');
  }
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

const formatOutcomeMessage = (state) => {
  if (!state || !state.done) {
    return '';
  }
  if (state.outcome === 'win') {
    return `🎉 Victory! Cleared the board in ${state.step} steps.`;
  }
  if (state.outcome === 'loss') {
    return `💥 Boom! A mine ended the run on step ${state.step}.`;
  }
  return `Game finished in ${state.step} steps.`;
};

const hideGameOver = () => {
  gameOverEl.classList.remove('visible');
  gameOverMessageEl.textContent = '';
};

const updateGameOver = (state) => {
  if (!state || !state.done) {
    hideGameOver();
    return;
  }
  gameOverMessageEl.textContent = formatOutcomeMessage(state);
  gameOverEl.classList.add('visible');
};

const renderBoard = (state) => {
  boardEl.classList.toggle('overlay-enabled', overlayEnabled);
  boardEl.classList.toggle('is-complete', Boolean(state.done));
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

      const face = document.createElement('span');
      face.classList.add('cell-face');
      cell.appendChild(face);

      if (revealed) {
        cell.classList.add('revealed');
        if (count > 0) {
          face.textContent = String(count);
          cell.classList.add(`count-${count}`);
        } else {
          face.innerHTML = '&nbsp;';
        }
        cell.disabled = true;
        cell.removeAttribute('title');
      } else {
        cell.classList.add('hidden');
        face.innerHTML = '&nbsp;';

        const overlay = document.createElement('span');
        overlay.classList.add('mine-overlay');
        overlay.setAttribute('aria-hidden', 'true');
        cell.appendChild(overlay);

        const label = document.createElement('span');
        label.classList.add('mine-label');
        label.setAttribute('aria-hidden', overlayEnabled ? 'false' : 'true');
        cell.appendChild(label);

        applyMineProbabilityDecor(cell, overlay, label, prob, r, c);

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
  updateGameOver(state);
  syncBoardOptions(state);
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

const startNewGame = async ({ preset = null, seed = null } = {}) => {
  if (isBusy) {
    return;
  }
  isBusy = true;
  updateStatus(null, preset ? 'Loading board…' : 'Resetting…');
  hideGameOver();
  try {
    const payload = {};
    if (preset) {
      payload.preset = preset;
    }
    if (seed !== null && seed !== undefined) {
      payload.seed = seed;
    }
    const nextState = await postJson('/api/new-game', payload);
    refresh(nextState);
  } catch (err) {
    console.error(err);
    updateStatus(currentState, 'Failed to reset game');
  } finally {
    isBusy = false;
  }
};

newGameBtn.addEventListener('click', () => startNewGame());
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

gameOverResetBtn.addEventListener('click', () => startNewGame());

bootstrap();

function syncBoardOptions(state) {
  if (!boardSelect || !state || !Array.isArray(state.preset_options)) {
    return;
  }

  const signature = state.preset_options
    .map((opt) => `${opt.id}:${opt.label}`)
    .join('|');

  if (signature !== presetCacheSignature) {
    presetCacheSignature = signature;
    boardSelect.innerHTML = '';
    state.preset_options.forEach((opt) => {
      const optionEl = document.createElement('option');
      optionEl.value = opt.id;
      optionEl.textContent = opt.label;
      boardSelect.appendChild(optionEl);
    });
  }

  const hasCurrentOption = state.preset_options.some((opt) => opt.id === state.preset);
  if (!hasCurrentOption && state.preset) {
    const optionEl = document.createElement('option');
    optionEl.value = state.preset;
    optionEl.textContent = state.preset_label || state.preset;
    boardSelect.appendChild(optionEl);
  }

  if (state.preset) {
    boardSelect.value = state.preset;
  }
  boardSelect.title = state.preset_label || '';
}

boardSelect?.addEventListener('change', (event) => {
  const nextPreset = event.target.value;
  if (!nextPreset || (currentState && currentState.preset === nextPreset)) {
    return;
  }
  startNewGame({ preset: nextPreset });
});
