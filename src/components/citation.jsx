import React from 'react';
import { render } from 'react-dom';
import UIkit from 'uikit';
import { LuClipboardCopy } from 'react-icons/lu';

const CopyButton = ({ text }) => {
  const copyToClipboard = () => {
    const tt = document.querySelector('.tooltip');
    navigator.clipboard.writeText(text).then(
      () => {
        UIkit.tooltip(tt, { title: 'Copied!' }).show();
        console.log('copied:');
      },
      (err) => {
        console.error('failed to copy text:', err);
      }
    );
  };
  return (
    <button
      className="tooltip uk-align-right"
      onClick={copyToClipboard}
      style={{
        border: 'none',
        background: 'transparent',
        color: '#333',
        cursor: 'pointer',
      }}
    >
      <LuClipboardCopy size={18} />
    </button>
  );
};

export default class Citation extends React.Component {
  constructor(props) {
    super(props);
  }

  render() {
    return (
      <div className="uk-section">
        <h2>Citation</h2>
        <pre className="uk-padding-small">
          <CopyButton text={this.props.bibtex} />
          <code id="bibtex">{this.props.bibtex}</code>
        </pre>
      </div>
    );
  }
}
