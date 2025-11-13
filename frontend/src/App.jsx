import { useMemo, useState } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const toDataUrl = (base64) => (base64 ? `data:image/png;base64,${base64}` : null);

function App() {
  const [file1, setFile1] = useState(null);
  const [file2, setFile2] = useState(null);
  const [file1Preview, setFile1Preview] = useState(null);
  const [file2Preview, setFile2Preview] = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  const createPreview = (file, setter) => {
    if (!file) {
      setter(null);
      return;
    }
    // Only create preview for image files, not PDFs
    if (file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setter(reader.result);
      };
      reader.readAsDataURL(file);
    } else {
      // For PDFs, we'll show them after comparison if needed
      setter(null);
    }
  };

  const handleFileChange = (event, setter, previewSetter) => {
    const selected = event.target.files?.[0];
    setter(selected || null);
    createPreview(selected, previewSetter);
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

  return (
    <div className="page">
      <div className="top-strip">
        <div className="site-brand">
          <span className="badge">OCI</span>
          <span className="brand-label">Oracle Cloud</span>
        </div>
      </div>

      <div className="page-content">
        <main className="workspace">
          <section className="workspace-card">
            <div className="workspace-header">
              <h2>Compare your drawings</h2>
              <p>
                Upload two CAD exports. We'll highlight matched components and differences for you.
              </p>
            </div>

            <form className="upload-form" onSubmit={handleSubmit}>
              <label>
                <span>First file</span>
                <input
                  type="file"
                  accept=".png,.jpg,.jpeg,.pdf"
                  onChange={(event) => handleFileChange(event, setFile1, setFile1Preview)}
                  disabled={isSubmitting}
                />
              </label>
              <label>
                <span>Second file</span>
                <input
                  type="file"
                  accept=".png,.jpg,.jpeg,.pdf"
                  onChange={(event) => handleFileChange(event, setFile2, setFile2Preview)}
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
                <h2>Results dashboard</h2>
                <p>
                  Oracle-style comparison cards show matched components and detailed difference maps in
                  seconds.
                </p>
              </div>
              <div className="images-grid">
                {highlightSrc && (
                  <figure>
                    <img src={highlightSrc} alt="Highlighted differences" />
                    <div className="figure-actions">
                      <a href={highlightSrc} target="_blank" rel="noopener noreferrer">
                        View
                      </a>
                      <a href={highlightSrc} download="highlighted_differences.png">
                        Download
                      </a>
                    </div>
                    <figcaption>Highlighted Differences</figcaption>
                  </figure>
                )}
                {file1Preview && (
                  <figure>
                    <img src={file1Preview} alt="Input image 1" />
                    <div className="figure-actions">
                      <a href={file1Preview} target="_blank" rel="noopener noreferrer">
                        View
                      </a>
                      <a href={file1Preview} download={file1?.name || "input_image_1"}>
                        Download
                      </a>
                    </div>
                    <figcaption>Input Image 1</figcaption>
                  </figure>
                )}
                {file2Preview && (
                  <figure>
                    <img src={file2Preview} alt="Input image 2" />
                    <div className="figure-actions">
                      <a href={file2Preview} target="_blank" rel="noopener noreferrer">
                        View
                      </a>
                      <a href={file2Preview} download={file2?.name || "input_image_2"}>
                        Download
                      </a>
                    </div>
                    <figcaption>Input Image 2</figcaption>
                  </figure>
                )}
              </div>

              <div className="matches">
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
              </div>
            </section>
          )}
        </main>
      </div>

      <footer>
        <span>Backend API: {API_URL}</span>
        <span>Inspired by Oracle Analytics design language</span>
      </footer>
    </div>
  );
}

export default App;

