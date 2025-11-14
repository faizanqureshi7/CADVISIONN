import { useEffect, useMemo, useState } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const toDataUrl = (base64) => (base64 ? `data:image/png;base64,${base64}` : null);

function App() {
  const [file1, setFile1] = useState(null);
  const [file2, setFile2] = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [modalImage, setModalImage] = useState(null);
  const [modalZoom, setModalZoom] = useState(1);

  const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

  useEffect(() => {
    if (!modalImage) {
      return;
    }
    const handleKeyDown = (event) => {
      if (event.key === "Escape") {
        setModalImage(null);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [modalImage]);

  const handleFileChange = (event, setter) => {
    const selected = event.target.files?.[0];
    setter(selected || null);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();

    if (!file1 || !file2) {
      setError("Please select both files before submitting.");
      return;
    }

    setError("");
    setIsSubmitting(true);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file1", file1, file1.name);
      formData.append("file2", file2, file2.name);

      const response = await fetch(`${API_URL}/compare`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        let detail = "Comparison failed.";
        try {
          const payload = await response.json();
          detail = payload.detail || detail;
        } catch (_) {
          detail = await response.text();
        }
        throw new Error(detail);
      }

      const payload = await response.json();
      setResult(payload);
    } catch (err) {
      setError(err.message || "An unexpected error occurred.");
    } finally {
      setIsSubmitting(false);
    }
  };

  const highlightSrc = useMemo(
    () => toDataUrl(result?.images?.highlighted_1),
    [result]
  );
  const input1Src = useMemo(() => toDataUrl(result?.images?.input_1), [result]);
  const input2Src = useMemo(() => toDataUrl(result?.images?.input_2), [result]);
  const summaryMarkup = useMemo(() => {
    const text = result?.ai_summary?.trim();
    if (!text) {
      return null;
    }

    const escapeHtml = (value) =>
      value
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");

    const bolded = escapeHtml(text).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    const bulletized = bolded.replace(/(^|\n)[*-]\s+/g, (match, newline) => {
      const prefix = newline ? "<br />" : "";
      return `${prefix}<span class="summary-bullet">•</span> `;
    });
    const html = bulletized.replace(/\n/g, "<br />");
    return { __html: html };
  }, [result]);

  const openModal = (src, alt) => {
    if (!src) {
      return;
    }
    setModalImage({ src, alt });
    setModalZoom(1);
  };

  const closeModal = () => setModalImage(null);

  const adjustZoom = (delta) => {
    setModalZoom((prev) => {
      const next = clamp(parseFloat((prev + delta).toFixed(2)), 1, 4);
      return next;
    });
  };

  const handleZoomInput = (event) => {
    const value = parseFloat(event.target.value);
    if (!Number.isNaN(value)) {
      setModalZoom(clamp(value, 1, 4));
    }
  };

  const handleWheelZoom = (event) => {
    event.preventDefault();
    const direction = event.deltaY < 0 ? 0.1 : -0.1;
    adjustZoom(direction);
  };

  return (
    <div className="page">
      <header className="top-strip">
        <div className="site-brand">
          <span className="brand-mark">OCI</span>
          <span className="brand-text">Oracle Cloud</span>
        </div>
        <span className="top-caption">CADVISION</span>
      </header>

      <div className="page-content">
        <section className="hero">
          <div className="hero-copy">
            <h1>CADVision: An AI-powered CAD Designs Comparison Tool</h1>
            <p>
              Inspired by the Oracle Cloud experience, CADVISION compares two drawing
              revisions, highlights differences, and keeps your review workflow simple.
            </p>
          </div>
          <div className="hero-visual">
            <div className="hero-visual-frame">
              <span className="hero-visual-label">Always Free Experience</span>
              <p>Upload two revisions and review differences in seconds.</p>
            </div>
          </div>
        </section>

        <main className="workspace">
          <section className="workspace-card">
            <div className="workspace-header">
              <h2>Compare your drawings</h2>
              <p>
                Choose two CAD exports (PNG, JPG, or PDF). CADVISION aligns components and highlights
                key changes automatically.
              </p>
            </div>

            <form className="upload-form" onSubmit={handleSubmit}>
              <label>
                <span>First file</span>
                <input
                  type="file"
                  accept=".png,.jpg,.jpeg,.pdf"
                  onChange={(event) => handleFileChange(event, setFile1)}
                  disabled={isSubmitting}
                />
              </label>
              <label>
                <span>Second file</span>
                <input
                  type="file"
                  accept=".png,.jpg,.jpeg,.pdf"
                  onChange={(event) => handleFileChange(event, setFile2)}
                  disabled={isSubmitting}
                />
              </label>

              <button className="compare-button" type="submit" disabled={isSubmitting}>
                {isSubmitting ? "Comparing..." : "Run Comparison"}
              </button>
            </form>

            {error && <div className="status error">{error}</div>}
          </section>

          {result && (
            <section className="results">
              <div className="results-header">
                <h2>Comparison results</h2>
                <p>Review the highlighted differences alongside the original inputs.</p>
                <div className="legend">
                  <span className="legend-item">
                    <span className="legend-swatch legend-add" />
                    Additions (green)
                  </span>
                  <span className="legend-item">
                    <span className="legend-swatch legend-del" />
                    Deletions (red)
                  </span>
                </div>
              </div>
              <div className="images-grid">
                
                {input1Src && (
                  <figure className="image-card">
                    <img src={input1Src} alt="Input drawing 1" />
                    <figcaption>Input Drawing 1</figcaption>
                    <div className="figure-actions">
                      <button
                        type="button"
                        className="view-button"
                        onClick={() => openModal(input1Src, "Input drawing 1")}
                      >
                        View
                      </button>
                      <a
                        href={input1Src}
                        download={file1?.name || "input_drawing_1.png"}
                        className="download-link"
                      >
                        Download
                      </a>
                    </div>
                  </figure>
                )}
                {input2Src && (
                  <figure className="image-card">
                    <img src={input2Src} alt="Input drawing 2" />
                    <figcaption>Input Drawing 2</figcaption>
                    <div className="figure-actions">
                      <button
                        type="button"
                        className="view-button"
                        onClick={() => openModal(input2Src, "Input drawing 2")}
                      >
                        View
                      </button>
                      <a
                        href={input2Src}
                        download={file2?.name || "input_drawing_2.png"}
                        className="download-link"
                      >
                        Download
                      </a>
                    </div>
                  </figure>
                )}
                {highlightSrc && (
                  <figure className="image-card">
                    <img src={highlightSrc} alt="Highlighted differences" />
                    <figcaption>Highlighted Differences</figcaption>
                    <div className="figure-actions">
                      <button
                        type="button"
                        className="view-button"
                        onClick={() => openModal(highlightSrc, "Highlighted differences")}
                      >
                        View
                      </button>
                      <a
                        href={highlightSrc}
                        download="highlighted_differences.png"
                        className="download-link"
                      >
                        Download
                      </a>
                    </div>
                  </figure>
                )}
              </div>
              <div className="summary-card">
                <div className="summary-card-header">
                  <div>
                    <p className="summary-eyebrow">AI-Generated Summary</p>
                    <h3>Revision Analysis</h3>
                  </div>
                </div>
                {summaryMarkup ? (
                  <div className="summary-content" dangerouslySetInnerHTML={summaryMarkup} />
                ) : (
                  <p className="summary-placeholder">
                    AI summary will appear here after a successful comparison.
                  </p>
                )}
              </div>

              {/* <div className="matches">
                <h3>Matched Components</h3>
                {Object.entries(result.matches || {}).length === 0 ? (
                  <p>No matches found.</p>
                ) : (
                  Object.entries(result.matches).map(([img1Index, matches]) => (
                    <div key={img1Index} className="match-group">
                      <strong>Image 1 component {img1Index}</strong>
                      <ul>
                        {matches.map((item) => (
                          <li key={`${img1Index}-${item.img2_index}`}>
                            Matches Image 2 component {item.img2_index} at{" "}
                            {(item.similarity * 100).toFixed(1)}% similarity
                          </li>
                        ))}
                      </ul>
                    </div>
                  ))
                )}
              </div> */}
            </section>
          )}
        </main>
      </div>


      {modalImage && (
        <div className="modal-backdrop" role="presentation" onClick={closeModal}>
          <div
            className="modal-dialog"
            role="dialog"
            aria-modal="true"
            aria-label={modalImage.alt}
            onClick={(event) => event.stopPropagation()}
          >
            <button type="button" className="modal-close" onClick={closeModal}>
              Close
            </button>
            <div className="modal-toolbar">
              <button type="button" onClick={() => adjustZoom(-0.1)}>
                −
              </button>
              <input
                type="range"
                min="1"
                max="4"
                step="0.1"
                value={modalZoom}
                onChange={handleZoomInput}
                aria-label="Zoom level"
              />
              <span>{Math.round(modalZoom * 100)}%</span>
              <button type="button" onClick={() => adjustZoom(0.1)}>
                +
              </button>
              <button type="button" onClick={() => setModalZoom(1)}>
                Reset
              </button>
            </div>
            <div className="modal-image-container" onWheel={handleWheelZoom}>
              <img
                src={modalImage.src}
                alt={modalImage.alt}
                style={{ transform: `scale(${modalZoom})` }}
              />
            </div>
            <p className="modal-caption">{modalImage.alt}</p>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;

