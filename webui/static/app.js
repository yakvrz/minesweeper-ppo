const boardEl = document.getElementById('board');
const statusEl = document.getElementById('status');
const newGameBtn = document.getElementById('new-game');
const boardPresetLabel = document.getElementById('board-preset-label');
const boardOutcomeEl = document.getElementById('board-outcome');
const statStepEl = document.getElementById('stat-step');
const statRevealedEl = document.getElementById('stat-revealed');
const statHiddenEl = document.getElementById('stat-hidden');
const gameOverEl = document.getElementById('game-over');
const gameOverMessageEl = document.getElementById('game-over-message');
const gameOverResetBtn = document.getElementById('game-over-reset');

let currentState = null;
let isBusy = false;
let flagBusy = false;

const formatPercent = (value) => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return '';
  }
  return `${Math.round(Number(value) * 100)}%`;
};

if (boardEl) {
  boardEl.addEventListener('contextmenu', (event) => event.preventDefault());
}

const setStatus = (mode, text) => {
  statusEl.textContent = text;
  statusEl.classList.remove('is-live', 'is-win', 'is-loss');
  statusEl.classList.add(mode);
};

const updateStatus = (state, message = null) => {
  if (message) {
    setStatus('is-live', message);
    return;
  }
  if (!state) {
    setStatus('is-live', 'Loading…');
    return;
  }
  if (state.done) {
    if (state.outcome === 'win') {
      setStatus('is-win', `Win in ${state.step} steps`);
    } else if (state.outcome === 'loss') {
      setStatus('is-loss', `Mine hit on step ${state.step}`);
    } else {
      setStatus('is-live', `Finished in ${state.step} steps`);
    }
  } else {
    setStatus('is-live', `Step ${state.step}`);
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
  boardEl.classList.toggle('board-compact', state.cols >= 12);
  boardEl.innerHTML = '';

  const nextMove = state.step > 0 ? state.next_move : null;
  const flags = state.flags || [];
  const mineProbs = state.mine_probabilities || [];

  for (let r = 0; r < state.rows; r += 1) {
    for (let c = 0; c < state.cols; c += 1) {
      const revealed = state.revealed[r][c];
      const count = state.counts[r][c];
      const flagged = flags[r] ? Boolean(flags[r][c]) : false;
      const mineProb = mineProbs[r] ? mineProbs[r][c] : null;

      const cell = document.createElement('button');
      cell.type = 'button';
      cell.dataset.row = String(r);
      cell.dataset.col = String(c);
      cell.dataset.flagged = flagged ? '1' : '0';
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
        cell.dataset.flagged = '0';
        cell.disabled = true;
        cell.removeAttribute('title');
        cell.classList.remove('next-move');
      } else {
        face.textContent = flagged ? '⚑' : '';
        if (!flagged) {
          face.innerHTML = '&nbsp;';
        }
        cell.classList.toggle('flagged', flagged);
        const baseAria = `Cell ${r + 1}, ${c + 1}`;
        cell.setAttribute('aria-label', flagged ? `${baseAria}: flagged` : baseAria);

        if (nextMove && nextMove.row === r && nextMove.col === c && !flagged) {
          cell.classList.add('next-move');
          const label = document.createElement('span');
          label.classList.add('mine-label', 'label-next');
          const percentText = formatPercent(mineProb);
          label.textContent = percentText ? `${percentText} NEXT` : 'NEXT';
          label.setAttribute('aria-hidden', 'false');
          cell.appendChild(label);
          const ariaText = percentText
            ? `${baseAria}: suggested next move (${percentText})`
            : `${baseAria}: suggested next move`;
          cell.setAttribute('aria-label', ariaText);
        }

        cell.addEventListener('contextmenu', onCellContextMenu);
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

const toggleFlag = async ({ row, col }) => {
  if (flagBusy || !currentState || currentState.done) {
    return;
  }
  flagBusy = true;
  try {
    const nextState = await postJson('/api/flag', { row, col });
    refresh(nextState);
  } catch (err) {
    console.error(err);
    updateStatus(currentState, 'Failed to toggle flag');
  } finally {
    flagBusy = false;
  }
};

const refresh = (state) => {
  currentState = state;
  renderBoard(state);
  updateStatus(state);
  updateHud(state);
  updateGameOver(state);
};

const onCellContextMenu = (event) => {
  event.preventDefault();
  const row = Number(event.currentTarget.dataset.row);
  const col = Number(event.currentTarget.dataset.col);
  if (!Number.isInteger(row) || !Number.isInteger(col)) {
    return;
  }
  if (!currentState || currentState.done) {
    return;
  }
  if (currentState.revealed[row][col]) {
    return;
  }
  toggleFlag({ row, col });
};

const onCellClick = async (event) => {
  if (isBusy) {
    return;
  }
  const row = Number(event.currentTarget.dataset.row);
  const col = Number(event.currentTarget.dataset.col);
  if (!Number.isInteger(row) || !Number.isInteger(col)) {
    return;
  }
  if (currentState && currentState.flags && currentState.flags[row] && currentState.flags[row][col]) {
    return;
  }
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

const startNewGame = async ({ seed = null } = {}) => {
  if (isBusy || flagBusy) {
    return;
  }
  isBusy = true;
  updateStatus(null, 'Resetting…');
  hideGameOver();
  try {
    const payload = {};
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

gameOverResetBtn.addEventListener('click', () => startNewGame());

const bootstrap = async () => {
  updateStatus(null, 'Loading…');
  try {
    const state = await fetchState();
    refresh(state);
  } catch (err) {
    console.error(err);
    updateStatus(null, 'Failed to load state');
  }
};

function updateHud(state) {
  if (!state) {
    return;
  }

  if (boardPresetLabel) {
    const detail = `${state.mine_count ?? '?'} mines · ${state.total_cells ?? state.rows * state.cols} tiles`;
    const label = state.board_label || `${state.rows}×${state.cols}`;
    boardPresetLabel.textContent = `${label} · ${detail}`;
  }
  if (boardOutcomeEl) {
    if (state.done) {
      boardOutcomeEl.textContent = state.outcome === 'win' ? 'Victory' : state.outcome === 'loss' ? 'Mine triggered' : 'Finished';
    } else {
      boardOutcomeEl.textContent = 'Live';
    }
  }
  if (statStepEl) {
    statStepEl.textContent = String(state.step ?? 0);
  }
  if (statRevealedEl) {
    statRevealedEl.textContent = String(state.revealed_count ?? 0);
  }
  if (statHiddenEl) {
    statHiddenEl.textContent = String(state.remaining_hidden ?? 0);
  }
}

bootstrap();
