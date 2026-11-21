// Presentation-only coordinates. A viewport never owns a simulation, a clock,
// an inspector request or a command handle. Both renderers and picking share it.
export const MIN_WORLD_ZOOM = 1;
export const MAX_WORLD_ZOOM = 32;
export const FIT_WORLD_VIEW = Object.freeze({ centerX: 0.5, centerY: 0.5, zoom: 1 });

const wrap = (value) => {
  const remainder = value % 1;
  const wrapped = remainder < 0 ? remainder + 1 : remainder;
  // Adding one to a negative rounding residual can itself round to one.
  return wrapped === 0 || wrapped === 1 ? 0 : wrapped;
};
const finite = (...values) => values.every(Number.isFinite);

function checkedView(view) {
  if (!view || !finite(view.centerX, view.centerY, view.zoom)
      || view.centerX < 0 || view.centerX >= 1 || view.centerY < 0 || view.centerY >= 1
      || view.zoom < MIN_WORLD_ZOOM || view.zoom > MAX_WORLD_ZOOM) {
    throw new Error("World view exceeds its finite zoom or toroidal bounds");
  }
  return view;
}

/** Prepare one transform per paint/pick, independent of the source publication. */
export function worldProjection(world, width, height, view = FIT_WORLD_VIEW) {
  checkedView(view);
  if (!finite(world?.width, world?.height, width, height)
      || world.width <= 0 || world.height <= 0 || width <= 0 || height <= 0) {
    throw new Error("World projection requires positive finite source and viewport dimensions");
  }
  return { worldWidth: world.width, worldHeight: world.height, width, height,
    centerX: view.centerX, centerY: view.centerY, zoom: view.zoom,
    periodX: width * view.zoom, periodY: height * view.zoom };
}

/** Visible copies are bounded to the nearest toroidal image and its eight neighbors. */
export function visitWorldCopies(position, projection, padding, visit) {
  if (!Array.isArray(position) || position.length !== 2 || !finite(...position, padding) || padding < 0) {
    throw new Error("World projection requires a finite position and visual radius");
  }
  const p = projection;
  const x = (wrap(position[0] / p.worldWidth - p.centerX + 0.5) - 0.5) * p.periodX + p.width / 2;
  const y = (wrap(position[1] / p.worldHeight - p.centerY + 0.5) - 0.5) * p.periodY + p.height / 2;
  for (let row = -1; row <= 1; row += 1) {
    const py = y + row * p.periodY;
    if (py + padding < 0 || py - padding > p.height) continue;
    for (let column = -1; column <= 1; column += 1) {
      const px = x + column * p.periodX;
      if (px + padding >= 0 && px - padding <= p.width) visit(px, py);
    }
  }
}

/** At zoom >= 1, at most four whole-world tiles cover the visible rectangle. */
export function visitWorldTiles(projection, visit) {
  const p = projection;
  const left = Math.floor(p.centerX - 0.5 / p.zoom);
  const right = Math.ceil(p.centerX + 0.5 / p.zoom);
  const top = Math.floor(p.centerY - 0.5 / p.zoom);
  const bottom = Math.ceil(p.centerY + 0.5 / p.zoom);
  for (let row = top; row < bottom; row += 1) {
    for (let column = left; column < right; column += 1) {
      visit((column - p.centerX) * p.periodX + p.width / 2,
        (row - p.centerY) * p.periodY + p.height / 2, p.periodX, p.periodY);
    }
  }
}

/** Immutable views let input target the pixels actually accepted by a renderer. */
export class WorldViewport {
  constructor() {
    this.view = FIT_WORLD_VIEW;
    this.revision = 0;
    this.source = null;
    this.followedUid = null;
    this.notice = "";
  }

  setView(centerX, centerY, zoom) {
    const next = { centerX: wrap(centerX), centerY: wrap(centerY), zoom };
    checkedView(next);
    if (Object.keys(next).every((key) => next[key] === this.view[key])) return false;
    this.view = Object.freeze(next);
    this.revision += 1;
    return true;
  }

  fit() {
    this.stopFollowing();
    this.notice = "";
    return this.setView(0.5, 0.5, 1);
  }

  stopFollowing(message = "") {
    const changed = this.followedUid !== null;
    this.followedUid = null;
    this.notice = message;
    return changed;
  }

  /** A new owner, epoch, saved-world view or world extent starts at a recoverable fit. */
  observe(frame, generation) {
    const world = frame?.snapshot?.world;
    worldProjection(world, 1, 1);
    if (generation === null || generation === undefined || typeof frame.identity?.sessionId !== "string"
        || frame.snapshot.epoch === null || frame.snapshot.epoch === undefined) {
      throw new Error("World view requires an explicit owner, epoch and source lifetime");
    }
    const source = JSON.stringify([generation, frame.identity.sessionId, frame.snapshot.epoch, world.width, world.height]);
    if (source !== this.source) {
      this.source = source;
      this.fit();
      // Source identity also invalidates an equal fit transform from another owner.
      this.revision += 1;
    }
    if (this.followedUid !== null) {
      const agent = frame.snapshot.agents.find((candidate) => candidate.uid === this.followedUid);
      if (!agent || !Array.isArray(agent.position) || agent.position.length !== 2 || !finite(...agent.position)) {
        this.stopFollowing("Follow ended: the selected creature is no longer present in this world.");
      } else {
        this.setView(agent.position[0] / world.width, agent.position[1] / world.height, this.view.zoom);
      }
    }
    return this.view;
  }

  follow(uid, snapshot) {
    if (typeof uid !== "string" || !/^(0|[1-9][0-9]{0,19})$/.test(uid) || BigInt(uid) >= (1n << 64n)) {
      throw new Error("Follow requires an exact selected creature UID");
    }
    const agent = snapshot?.agents.find((candidate) => candidate.uid === uid);
    if (!agent || !Array.isArray(agent.position) || agent.position.length !== 2 || !finite(...agent.position)) {
      throw new Error("Select a creature present in the displayed world before following it");
    }
    worldProjection(snapshot.world, 1, 1);
    this.followedUid = uid;
    this.notice = "";
    this.setView(agent.position[0] / snapshot.world.width, agent.position[1] / snapshot.world.height, this.view.zoom);
  }

  /** Keep the pointed-to visible world location fixed, including across a seam. */
  zoomBy(factor, x = 0.5, y = 0.5, paintedView = this.view) {
    checkedView(paintedView);
    if (!finite(factor, x, y) || factor <= 0 || x < 0 || x > 1 || y < 0 || y > 1) return false;
    const zoom = Math.max(MIN_WORLD_ZOOM, Math.min(MAX_WORLD_ZOOM, this.view.zoom * factor));
    if (zoom === this.view.zoom) return false;
    // Following keeps the subject centered; a manual drag explicitly releases it.
    if (this.followedUid !== null) { x = 0.5; y = 0.5; }
    const centerX = paintedView.centerX + (x - 0.5) / paintedView.zoom - (x - 0.5) / zoom;
    const centerY = paintedView.centerY + (y - 0.5) / paintedView.zoom - (y - 0.5) / zoom;
    this.notice = "";
    return this.setView(centerX, centerY, zoom);
  }

  /** Drag relative to its immutable starting paint, not an unpainted queued view. */
  panFrom(paintedView, deltaX, deltaY) {
    checkedView(paintedView);
    if (!finite(deltaX, deltaY)) return false;
    this.stopFollowing();
    return this.setView(paintedView.centerX - deltaX / paintedView.zoom,
      paintedView.centerY - deltaY / paintedView.zoom, paintedView.zoom);
  }
}

/** One primary pointer; small click jitter never moves the camera or loses selection. */
export class WorldDrag {
  constructor() { this.cancel(); }

  begin(pointerId, x, y, view, width = null, height = null) {
    if (this.active || !finite(x, y)) return false;
    if ((width !== null || height !== null) && (!finite(width, height) || width <= 0 || height <= 0)) return false;
    checkedView(view);
    this.suppressClick = false;
    this.active = { pointerId, x, y, view, width, height, dragged: false };
    return true;
  }

  move(pointerId, x, y, width = this.active?.width, height = this.active?.height) {
    const active = this.active;
    if (!active || active.pointerId !== pointerId || !finite(x, y)) return null;
    if (width !== active.width || height !== active.height) { this.cancel(true); return null; }
    const deltaX = x - active.x, deltaY = y - active.y;
    if (!active.dragged && Math.hypot(deltaX, deltaY) < 4) return null;
    active.dragged = true;
    return { view: active.view, deltaX, deltaY };
  }

  end(pointerId) {
    if (!this.active || this.active.pointerId !== pointerId) return false;
    this.suppressClick = this.active.dragged;
    this.active = null;
    return true;
  }

  cancel(suppressClick = false) { this.active = null; this.suppressClick = suppressClick; }

  consumeClick() {
    const suppress = this.suppressClick;
    this.suppressClick = false;
    return suppress;
  }
}
