// Preserve every source scene; append only the final source-order step.
const originalFlowBuild = flowBuild;
flowBuild = function () {
  originalFlowBuild();
  flowAdd(5, 'source-sort', 'Sort theo thứ tự câu trong input', 'Sắp tăng dần theo tài liệu và vị trí câu gốc, giữ nguyên tập câu đã chọn.');
};
const originalFlowContent = flowRenderContent;
flowRenderContent = function (item) {
  if (item.type === 'source-sort') {
    return window.SourceSortView.render(DATA.sorting) + '<h3 class="sorted-title">Bản tóm tắt sau Sort</h3><ol class="final-list sorted-final">' + DATA.sorting.after.map(sentence => '<li><small>' + esc(sentence.doc_label) + ' · câu ' + sentence.sourcePosition + ' · ' + esc(sentence.id) + '</small>' + esc(sentence.text) + '</li>').join('') + '</ol>';
  }
  return originalFlowContent(item);
};


// Finish the current scene immediately and keep its result visible.
function skipScene() {
  const item = flowSteps[flowIndex];
  flowStop();
  kmLessonStopTimer();
  if (item.type === 'km-frame' && kmLessonState) {
    Object.assign(kmLessonState, {running: false, cursor: DATA.sentences.length - 1, pointPhase: 'assigned', phase: 'complete'});
    kmLessonRefresh();
  } else if (item.type === 'km-init' && kmInitState) {
    Object.assign(kmInitState, {running: false, cursor: (kmInitState.frame.centers || []).length - 1});
    kmInitRefresh();
  } else if (item.type === 'km-move' && kmMoveState) {
    const centers = kmMoveState.frame.centers_after || kmMoveState.frame.centers || [];
    Object.assign(kmMoveState, {running: false, clusterIndex: centers.length - 1});
    kmMoveRefresh();
    $('km-move-stage').innerHTML = kmSvgBase(kmMoveState.frame, 'complete', DATA.sentences.length - 1);
    $('km-move-status').textContent = '✓ ĐÃ CẬP NHẬT TOÀN BỘ ' + centers.length + ' TÂM';
  } else if (item.type === 'km-animation') {
    const run = kmAnimationRun();
    kmAnimIndex = Math.max(0, (run.history || DATA.kmeans.history || []).length - 1);
    renderKmGraphOnly();
  }
  const button = $('scene-skip');
  if (button) { button.disabled = true; button.textContent = '✓ Đã hoàn tất'; }
}
const originalFlowRender = flowRender;
flowRender = function () {
  originalFlowRender();
  const item = flowSteps[flowIndex];
  if (!item) return;
  const toolbar = $('flow-content').querySelector('.km-lesson-toolbar, .km-graph-toolbar');
  if (toolbar && ['km-frame', 'km-init', 'km-move', 'km-animation'].includes(item.type)) {
    const button = document.createElement('button');
    button.type = 'button';
    button.id = 'scene-skip';
    button.textContent = '⏭ Skip · Hoàn tất bước';
    button.addEventListener('click', skipScene);
    toolbar.appendChild(button);
  }
};
$('flow-skip-group').onclick = () => {
  flowStop(); kmLessonStopTimer();
  const group = flowSteps[flowIndex].group;
  let last = flowIndex;
  while (last + 1 < flowSteps.length && flowSteps[last + 1].group === group) last++;
  flowGo(last);
};
$('flow-skip-end').onclick = () => {
  flowStop(); kmLessonStopTimer();
  flowGo(flowSteps.length - 1);
};
