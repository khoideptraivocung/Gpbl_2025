
import { NextResponse } from 'next/server';

/**
 * GETリクエストを処理して、Raspberry Piのデータを取得し、クライアントに返します。
 */
export async function GET() {
  // Raspberry Piで起動したサーバーのURL
  // <RASPBERRY_PI_IP_ADDRESS> を先ほどメモしたIPアドレスに置き換えてください。
  const RASPBERRY_PI_URL = 'http://10.136.132.35:8000/data';

  try {
    // Raspberry Piのサーバーにデータをリクエスト
    const response = await fetch(RASPBERRY_PI_URL, { cache: 'no-store' });

    // レスポンスが正常でない場合はエラーを投げる
    if (!response.ok) {
      throw new Error(`Failed to fetch from Raspberry Pi: ${response.status} ${response.statusText}`);
    }

    // レスポンスのJSONをパース
    const data = await response.json();

    // 取得したデータをフロントエンドに返す
    return NextResponse.json(data);

  } catch (error) {
    // エラーハンドリング
    console.error("Failed to fetch data from Raspberry Pi:", error);

    // エラーが発生した場合は、フロントエンドにエラーメッセージを返す
    // バックエンドのコンソールに詳細なエラーが表示されます
    return NextResponse.json(
      { message: "Error fetching data from Raspberry Pi. Is the Pi server running and reachable?" },
      { status: 500 }
    );
  }
}
