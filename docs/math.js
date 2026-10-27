// Render local TeX while retaining readable fallback text if rendering fails.
document.querySelectorAll('[data-tex]').forEach((element) => {
  try {
    katex.render(element.dataset.tex, element, {
      displayMode: element.classList.contains('equation'),
      throwOnError: true,
    });
  } catch (error) {
    console.error('Could not render equation:', error);
  }
});
