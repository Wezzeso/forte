import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.tsx'
import { initializeRuntimeConfig } from './runtime-config'
import './styles.css'
import './zimran.css'

async function start() {
  await initializeRuntimeConfig()
  ReactDOM.createRoot(document.getElementById('root')!).render(
    <React.StrictMode>
      <App />
    </React.StrictMode>,
  )
}

void start()
